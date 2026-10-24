"""Source-bound tensor-product measurements (Python 3.10, PyTorch 2.11).

The controller imports no numerical library and samples the child process, never
the whole GPU. A run records measurements; only an independent numerical check
and paired comparison can establish acceptance. All tensor inputs are bounded
and reused, and each step releases its autograd graph.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


SCHEMA = 1
FAMILIES = ("tri-unweighted", "tri-weighted", "uvuv-scalar-left", "uvuv-scalar-right", "uvuv-scalar-output")
# Keep the existing FP32 gate and make FP64 at least as strict as the library tests.
NUMERICAL_TOLERANCES = {
    "torch.float32": {"rtol": 1.3e-6, "atol": 1e-5},
    "torch.float64": {"rtol": 1e-9, "atol": 1e-10},
}


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_files(root):
    root = Path(root).resolve()
    files = {p.relative_to(root).as_posix(): digest_file(p) for p in sorted((root / "e3nn").rglob("*.py"))}
    if "e3nn/__init__.py" not in files or len(files) < 20:
        raise ValueError("Incomplete e3nn source tree")
    return files


def write_json(path, value, replace=False):
    """Publish complete evidence atomically without overwriting historical runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            if replace:
                # CPython's compact one-shot encoder avoids the recursive Python
                # encoder cycles that otherwise accumulate between memory samples.
                handle.write(json.dumps(value, sort_keys=True, allow_nan=False))
            else:
                json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        try:
            Path(tmp).unlink()
        except FileNotFoundError:
            pass


def snapshot(args):
    root = args.source_root.resolve()
    revision_root = (args.revision_root or root).resolve()
    files = source_files(root)
    git_top = subprocess.check_output(["git", "-C", str(revision_root), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(git_top).resolve() != revision_root:
        raise ValueError("Revision root must be the T-e3nn checkout root")
    head = subprocess.check_output(["git", "-C", str(revision_root), "rev-parse", "HEAD"], text=True).strip()
    if len(head) != 40:
        raise ValueError("Expected a real Git revision")
    dirty = subprocess.check_output(["git", "-C", str(revision_root), "status", "--porcelain", "--", "e3nn"], text=True)
    write_json(
        args.output,
        {
            "schema": SCHEMA,
            "root": str(root),
            "revision_root": str(revision_root),
            "head": head,
            "dirty": dirty.splitlines(),
            "files": files,
            "tree_sha256": digest_json(files),
            "driver_sha256": digest_file(__file__),
        },
    )


def validate_source(root, manifest):
    value = json.loads(Path(manifest).read_text())
    files = source_files(root)
    if (
        value.get("schema") != SCHEMA
        or value.get("root") != str(Path(root).resolve())
        or value.get("files") != files
        or value.get("tree_sha256") != digest_json(files)
        or value.get("driver_sha256") != digest_file(__file__)
        or len(value.get("head", "")) != 40
    ):
        raise ValueError("Source/driver manifest mismatch")
    return value


def assert_imports(root):
    package = Path(root).resolve() / "e3nn"
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name == "e3nn" or name.startswith("e3nn."):
            location = getattr(module, "__file__", None)
            if location is None:
                continue
            path = Path(location).resolve()
            try:
                path.relative_to(package)
            except ValueError:
                raise ValueError("Foreign e3nn import: " + str(path)) from None
            loaded[name] = str(path)
    if "e3nn" not in loaded:
        raise ValueError("e3nn was not imported")
    return loaded


def process_identity(pid):
    text = Path("/proc/{}/stat".format(pid)).read_text()
    # comm may contain spaces and parentheses; starttime is field 22.
    return {"pid": pid, "start_ticks": int(text[text.rfind(")") + 2 :].split()[19])}


def host_sample(pid):
    values = {}
    for line in Path("/proc/{}/status".format(pid)).read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, number, unit = line.split()
            if unit != "kB" or int(number) <= 0:
                raise ValueError("Invalid process memory")
            values[key[:-1]] = int(number) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise ValueError("Live child is missing RSS evidence")
    return values


def normalize_gpu_uuid(value):
    """Torch and NVML differ only in the optional GPU- prefix on physical UUIDs."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value) is None
    ):
        raise ValueError("Invalid physical GPU UUID: {!r}".format(value))
    return "GPU-" + str(uuid.UUID(value[4:] if value.startswith("GPU-") else value))


class GPUProcessEvidenceError(ValueError):
    """Keep the actual NVML rows when a process ownership assertion fails."""

    def __init__(self, identity, expected_uuid, rows):
        self.diagnostics = {"process": identity, "expected_uuid": expected_uuid, "observed_rows": rows}
        super().__init__(
            "Missing, duplicated, or wrong-UUID GPU process memory: " + json.dumps(self.diagnostics, sort_keys=True)
        )


def gpu_rows():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            raise ValueError("Unparseable GPU process memory: " + line)
        rows.append(
            {
                "pid": int(fields[0]),
                "uuid": normalize_gpu_uuid(fields[1]),
                "uuid_raw": fields[1],
                "bytes": int(fields[2]) * 1024 * 1024,
                "raw_line": line,
            }
        )
    return rows


def sample_process(identity, gpu_uuid=None):
    if process_identity(identity["pid"]) != identity:
        raise ValueError("Child PID was replaced")
    sample = {"monotonic_ns": time.monotonic_ns(), **identity, **host_sample(identity["pid"])}
    if gpu_uuid is not None:
        expected_uuid = normalize_gpu_uuid(gpu_uuid)
        rows = [row for row in gpu_rows() if row["pid"] == identity["pid"]]
        if len(rows) != 1 or rows[0]["uuid"] != expected_uuid or rows[0]["bytes"] <= 0:
            raise GPUProcessEvidenceError(identity, expected_uuid, rows)
        sample["gpu_bytes"] = rows[0]["bytes"]
        sample["gpu_uuid"] = rows[0]["uuid"]
        sample["gpu_row"] = rows[0]
    if process_identity(identity["pid"]) != identity:
        raise ValueError("Child PID changed while sampling")
    return sample


def handshake(control, phase, gpu_uuid=None):
    """Boundaries guarantee samples even when the timed block is very short."""
    write_json(control / "phase.json", {"phase": phase, "uuid": gpu_uuid}, replace=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        ack = control / "ack.json"
        if ack.exists() and json.loads(ack.read_text()).get("phase") == phase:
            return
        time.sleep(0.01)
    raise TimeoutError("Controller did not acknowledge " + phase)


def make_workload(args, torch, o3):
    mul, ell = args.mul, args.ell
    tri = args.family.startswith("tri-")
    l1 = 0 if args.family == "uvuv-scalar-left" else ell
    l2 = 0 if args.family == "uvuv-scalar-right" else ell
    if tri:
        outputs = list(range(2 * ell + 1)) if args.outputs == "all" else [ell]
        mode = "uvu<v" if args.family == "tri-unweighted" else "u<vw"
        out_mul = mul * (mul - 1) // 2 if mode == "uvu<v" else mul
    else:
        outputs = [0 if args.family == "uvuv-scalar-output" else ell]
        mode, out_mul = "uvuv", mul * mul
    weighted = args.family == "tri-weighted" or (not tri and args.weights != "none")
    shared = args.weights != "unshared"
    if args.family == "tri-weighted" and args.weights == "none":
        raise ValueError("u<vw requires weights")
    if args.alias_inputs and l1 != l2:
        raise ValueError("Aliased inputs require equal input irreps")
    irreps1, irreps2 = o3.Irreps([(mul, (l1, 1, -1))]), o3.Irreps([(mul, (l2, 1, -1))])
    irreps_out = o3.Irreps([(out_mul, (ell_out, 1, 1)) for ell_out in outputs])
    module = o3.TensorProduct(
        irreps1,
        irreps2,
        irreps_out,
        [(0, 0, i, mode, weighted) for i in range(len(outputs))],
        internal_weights=False,
        shared_weights=shared,
    )
    dtype = getattr(torch, args.dtype)
    module = module.to(device=args.device, dtype=dtype)
    # A private CPU generator makes inputs/weights independent of construction RNG use.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    tensors, hashes = {}, {}

    def random(name, shape, grad=False):
        tensor = torch.randn(shape, dtype=dtype, generator=generator)
        hashes[name] = {"sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(), "shape": list(shape)}
        tensors[name] = tensor.to(args.device).requires_grad_(grad)
        return tensors[name]

    grad = args.workload != "inference"
    x = random("input1", (args.batch, irreps1.dim), grad)
    y = x if args.alias_inputs else random("input2", (args.batch, irreps2.dim), grad)
    if args.alias_inputs:
        hashes["input2"] = {**hashes["input1"], "alias_of": "input1"}
    w = random("weight", (module.weight_numel,) if shared else (args.batch, module.weight_numel), grad) if weighted else None
    cotangent = random("output_cotangent", (args.batch, irreps_out.dim)) if grad else None
    variables = [x] if args.alias_inputs else [x, y]
    input_count = len(variables)
    targets = (
        [random("force_target_{}".format(i), tuple(t.shape)) for i, t in enumerate(variables)]
        if args.workload == "double-backward"
        else []
    )
    if w is not None:
        variables.append(w)
    description = {
        "family": args.family,
        "batch": args.batch,
        "mul": mul,
        "ell": ell,
        "outputs": args.outputs,
        "weights": args.weights,
        "alias_inputs": args.alias_inputs,
        "workload": args.workload,
        "seed": args.seed,
        "dtype": args.dtype,
        "irreps_in1": str(irreps1),
        "irreps_in2": str(irreps2),
        "irreps_out": str(irreps_out),
        "instructions": [list(ins) for ins in module.instructions],
        "weight_numel": module.weight_numel,
        "input_digests": hashes,
    }

    def step():
        if args.workload == "inference":
            with torch.no_grad():
                return (module(x, y, w).detach(),)
        output = module(x, y, w)
        energy = (output.tanh() * cotangent).sum() / math.sqrt(max(output.numel(), 1))
        if args.workload == "backward":
            gradients = torch.autograd.grad(energy, variables)
            return (output.detach(),) + tuple(value.detach() for value in gradients)
        forces = torch.autograd.grad(energy, variables[:input_count], create_graph=True)
        loss = sum((force - target).square().mean() for force, target in zip(forces, targets))
        gradients = torch.autograd.grad(loss, variables)
        return (output.detach(),) + tuple(value.detach() for value in forces) + tuple(value.detach() for value in gradients)

    return module, step, description


def child(args):
    before = validate_source(args.source_root, args.manifest)
    if any(name == "e3nn" or name.startswith("e3nn.") for name in sys.modules):
        raise ValueError("e3nn was imported before source activation")
    sys.path.insert(0, str(args.source_root.resolve()))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import e3nn
    from e3nn import o3

    assert_imports(args.source_root)
    if not torch.__version__.startswith("2.11."):
        raise ValueError("This measurement protocol requires pinned torch 2.11")
    torch.set_num_threads(1)
    torch.set_default_dtype(getattr(torch, args.dtype))
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    runtime = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "e3nn": e3nn.__version__,
        "deterministic_algorithms": True,
        "tf32": False,
        "float32_matmul_precision": "highest",
        "threads": torch.get_num_threads(),
    }
    gpu_uuid = None
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA device is required")
        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        if os.environ.get("GPU_TYPE") and os.environ["GPU_TYPE"] not in prop.name:
            raise ValueError("Allocated GPU name disagrees with GPU_TYPE")
        gpu_uuid = normalize_gpu_uuid(str(prop.uuid))
        runtime["gpu"] = {
            "uuid": gpu_uuid,
            "uuid_raw": str(prop.uuid),
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            "total_memory": prop.total_memory,
        }
    setup_start = time.perf_counter()
    module, step, workload = make_workload(args, torch, o3)
    if args.device == "cuda":
        torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    handshake(args.control, "cold", gpu_uuid)
    warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        values = step()
        del values
    if args.device == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start
    cold_allocator = allocator(torch, args.device)
    handshake(args.control, "steady", gpu_uuid)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    blocks = []
    if args.mode == "profile":
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profile:
            for _ in range(args.profile_steps):
                values = step()
                del values
            if args.device == "cuda":
                torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.control / "trace.json"))
        (args.control / "profile.txt").write_text(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    else:
        for index in range(args.blocks):
            if args.device == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
            begin = time.perf_counter()
            for _ in range(args.steps):
                values = step()
                del values
            if args.device == "cuda":
                end.record()
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            blocks.append(
                {
                    "index": index,
                    "steps": args.steps,
                    "wall_seconds": elapsed,
                    "cuda_seconds": start.elapsed_time(end) / 1000 if args.device == "cuda" else None,
                }
            )
    steady_allocator = allocator(torch, args.device)
    handshake(args.control, "complete", gpu_uuid)
    values = step()
    if any(not torch.isfinite(value).all().item() for value in values):
        raise FloatingPointError("Nonfinite output or derivatives")
    numerical = {
        "finite": True,
        "tensor_shapes": [list(value.shape) for value in values],
        "l2_norms": [float(value.norm()) for value in values],
    }
    if args.tensor_output:
        if args.tensor_output.exists():
            raise FileExistsError(str(args.tensor_output))
        torch.save(tuple(value.cpu() for value in values), args.tensor_output)
        numerical["artifact"] = {"path": str(args.tensor_output.resolve()), "sha256": digest_file(args.tensor_output)}
    del values
    loaded = assert_imports(args.source_root)
    after = validate_source(args.source_root, args.manifest)
    if before != after:
        raise ValueError("Source identity changed during run")
    graph = getattr(module, "_compiled_main_left_right", None)
    (args.control / "generated_graph.txt").write_text(str(getattr(graph, "code", graph)))
    write_json(
        args.control / "child.json",
        {
            "schema": SCHEMA,
            "status": "RECORDED",
            "passed": False,
            "source": before,
            "imports": loaded,
            "runtime": runtime,
            "workload": workload,
            "workload_sha256": digest_json(workload),
            "pid": os.getpid(),
            "mode": args.mode,
            "warmup": args.warmup,
            "blocks": blocks,
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "cold_allocator": cold_allocator,
            "steady_allocator": steady_allocator,
            "numerical": numerical,
        },
    )


def allocator(torch, device):
    if device != "cuda":
        return None
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def summarize_samples(samples, phase):
    values = [sample for sample in samples if sample["phase"] == phase]
    if not values:
        raise ValueError("Missing memory samples for " + phase)
    result = {
        "samples": len(values),
        "host_peak_bytes": max(v["VmRSS"] for v in values),
        "host_median_bytes": statistics.median(v["VmRSS"] for v in values),
        "host_hwm_bytes": max(v["VmHWM"] for v in values),
    }
    if "gpu_bytes" in values[0]:
        result.update(
            gpu_peak_bytes=max(v["gpu_bytes"] for v in values),
            gpu_median_bytes=statistics.median(v["gpu_bytes"] for v in values),
        )
    return result


def run(args):
    validate_source(args.source_root, args.manifest)
    if args.device == "cuda" and any(row["pid"] == os.getpid() for row in gpu_rows()):
        raise ValueError("Measurement controller already owns a CUDA context")
    args.output.mkdir(parents=True, exist_ok=False)
    control = args.output.resolve()
    command = [sys.executable, str(Path(__file__).resolve()), "_child", "--control", str(control)]
    for name in (
        "source_root",
        "manifest",
        "family",
        "batch",
        "mul",
        "ell",
        "outputs",
        "weights",
        "workload",
        "seed",
        "dtype",
        "device",
        "mode",
        "warmup",
        "blocks",
        "steps",
        "profile_steps",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.alias_inputs:
        command.append("--alias-inputs")
    if args.tensor_output:
        command.extend(["--tensor-output", str(args.tensor_output.resolve())])
    environment = os.environ.copy()
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(control / "inductor")
    environment["TRITON_CACHE_DIR"] = str(control / "triton")
    environment["OMP_NUM_THREADS"] = "1"
    samples, errors, sampling_errors, identity = [], [], [], None
    started = time.monotonic()
    with (control / "stdout.log").open("w") as stdout, (control / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        try:
            identity = process_identity(process.pid)
            complete = False
            while process.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("Measurement child exceeded deadline")
                phase_file = control / "phase.json"
                if phase_file.exists() and not complete:
                    phase = json.loads(phase_file.read_text())
                    sample = sample_process(identity, phase["uuid"])
                    # The completion handshake closes the window before CUDA teardown.
                    sample["phase"] = "steady" if phase["phase"] == "complete" else phase["phase"]
                    samples.append(sample)
                    write_json(control / "ack.json", {"phase": phase["phase"]}, replace=True)
                    complete = phase["phase"] == "complete"
                time.sleep(args.sample_interval)
            if process.returncode != 0:
                raise RuntimeError("Child exited with status {}".format(process.returncode))
            if not complete:
                raise ValueError("Child did not close the measurement window")
        except Exception as error:
            errors.append(str(error))
            if isinstance(error, GPUProcessEvidenceError):
                sampling_errors.append(error.diagnostics)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    write_json(control / "samples.json", samples)
    report = {
        "schema": SCHEMA,
        "status": "INVALID",
        "passed": False,
        "run_complete": False,
        "run_id": str(uuid.uuid4()),
        "process": identity,
        "errors": errors,
        "sampling_errors": sampling_errors,
        "returncode": process.returncode,
        "allocation": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "controller_pid": os.getpid(),
        "command": command,
        "driver_sha256": digest_file(__file__),
        "started_monotonic_ns": int(started * 1e9),
        "finished_monotonic_ns": time.monotonic_ns(),
        "measurement": {
            "sample_interval_seconds": args.sample_interval,
            "timing": "synchronized_wall_and_cuda_event",
            "sampler": "independent_parent_process",
        },
    }
    if not errors:
        report.update(json.loads((control / "child.json").read_text()))
        report.update(
            run_complete=True,
            memory={
                "scope": "child_pid_and_uuid",
                "window": "post_setup_through_measurement",
                "cold": summarize_samples(samples, "cold"),
                "steady": summarize_samples(samples, "steady"),
                "samples_sha256": digest_file(control / "samples.json"),
            },
        )
    write_json(control / "report.json", report)
    print(json.dumps({"report": str(control / "report.json"), "status": report["status"]}))
    if errors:
        raise RuntimeError("; ".join(errors))


def compare_tensors(args):
    import torch

    reports = [json.loads(path.read_text()) for path in (args.baseline, args.candidate)]
    for report in reports:
        validate_report(report, formal=False)
    if reports[0]["workload"] != reports[1]["workload"] or reports[0]["runtime"] != reports[1]["runtime"]:
        raise ValueError("Numerical workload/runtime mismatch")
    tensors = []
    for report in reports:
        artifact = report["numerical"]["artifact"]
        if digest_file(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Numerical artifact changed")
        tensors.append(torch.load(artifact["path"], map_location="cpu", weights_only=True))
    if len(tensors[0]) != len(tensors[1]):
        raise ValueError("Derivative participation mismatch")
    differences = []
    tolerances = []
    for baseline, candidate in zip(*tensors):
        dtype = str(baseline.dtype)
        if dtype not in NUMERICAL_TOLERANCES or candidate.dtype != baseline.dtype:
            raise ValueError("Numerical comparison requires matching float32 or float64 tensors")
        tolerance = NUMERICAL_TOLERANCES[dtype]
        torch.testing.assert_close(candidate, baseline, **tolerance)
        differences.append(float((candidate - baseline).abs().max()) if baseline.numel() else 0.0)
        tolerances.append({"dtype": dtype, **tolerance})
    write_json(
        args.output,
        {
            "status": "PASS",
            "kind": "numerical_only",
            "tolerance": "explicit_dtype_tolerances_v1",
            "tensor_tolerances": tolerances,
            "comparator_sha256": digest_file(__file__),
            "reports": [{"path": str(p.resolve()), "sha256": digest_file(p)} for p in (args.baseline, args.candidate)],
            "max_abs_differences": differences,
        },
    )


def validate_report(report, formal=True):
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "RECORDED"
        or report.get("passed") is not False
        or report.get("run_complete") is not True
        or report.get("errors")
        or report.get("returncode") != 0
        or report.get("mode") != "measure"
    ):
        raise ValueError("Incomplete or invalid measurement report")
    if report.get("workload_sha256") != digest_json(report["workload"]):
        raise ValueError("Workload digest mismatch")
    source = report["source"]
    if (
        source.get("tree_sha256") != digest_json(source["files"])
        or len(source["files"]) < 20
        or source.get("driver_sha256") != report.get("driver_sha256")
    ):
        raise ValueError("Incomplete source evidence")
    if not report["numerical"].get("finite"):
        raise ValueError("Missing finite numerical probe")
    if formal:
        if not report.get("allocation") or not report["runtime"].get("gpu", {}).get("uuid"):
            raise ValueError("Formal comparison requires a Slurm GPU allocation")
        if report["warmup"] < 20 or len(report["blocks"]) < 5 or any(b["steps"] < 200 for b in report["blocks"]):
            raise ValueError("Insufficient warmup or measured work")
    for block in report["blocks"]:
        if block["wall_seconds"] <= 0 or not math.isfinite(block["wall_seconds"]):
            raise ValueError("Invalid timing")
    if report["process"]["pid"] != report["pid"] or report["memory"]["scope"] != "child_pid_and_uuid":
        raise ValueError("Memory process identity mismatch")
    if report["finished_monotonic_ns"] <= report["started_monotonic_ns"]:
        raise ValueError("Invalid execution time window")


def per_step_wall(report):
    return statistics.median(block["wall_seconds"] / block["steps"] for block in report["blocks"])


def compare_pairs(args):
    """Consume ordered A/B records; collection alone never implies acceptance."""
    protocol = json.loads(args.pairs.read_text())
    pairs = protocol["pairs"]
    if len(pairs) < 5:
        raise ValueError("At least five independent pairs are required")
    identities, ratios, sources, common = set(), [], {}, None
    bindings = []
    previous_end = 0
    for index, pair in enumerate(pairs):
        expected_order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        if pair.get("order") != expected_order:
            raise ValueError("Pairs must alternate AB/BA")
        reports = {}
        for arm in expected_order:
            path = Path(pair[arm])
            report = json.loads(path.read_text())
            validate_report(report)
            if report["started_monotonic_ns"] <= previous_end:
                raise ValueError("Declared AB/BA order disagrees with execution windows")
            previous_end = report["finished_monotonic_ns"]
            samples_path = path.parent / "samples.json"
            if digest_file(samples_path) != report["memory"]["samples_sha256"]:
                raise ValueError("Memory samples changed")
            samples = json.loads(samples_path.read_text())
            if any(
                sample["pid"] != report["pid"]
                or sample["start_ticks"] != report["process"]["start_ticks"]
                or sample.get("gpu_uuid") != report["runtime"]["gpu"]["uuid"]
                for sample in samples
            ):
                raise ValueError("Raw memory process/UUID identity mismatch")
            if any(summarize_samples(samples, phase) != report["memory"][phase] for phase in ("cold", "steady")):
                raise ValueError("Memory summary disagrees with raw samples")
            identity = (report["hostname"], report["process"]["pid"], report["process"]["start_ticks"])
            if identity in identities:
                raise ValueError("A process was reused across measurements")
            identities.add(identity)
            signature = (
                report["allocation"],
                report["hostname"],
                report["runtime"],
                report["workload"],
                report["warmup"],
                [block["steps"] for block in report["blocks"]],
                report["memory"]["window"],
                report["driver_sha256"],
                report["measurement"],
            )
            if common is not None and signature != common:
                raise ValueError("Paired allocation/runtime/workload mismatch")
            common = signature
            if arm in sources and sources[arm] != report["source"]:
                raise ValueError("Source drift within arm")
            sources[arm] = report["source"]
            reports[arm] = report
            bindings.append({"path": str(path.resolve()), "sha256": digest_file(path)})
        baseline, candidate = reports["baseline"], reports["candidate"]
        ratios.append(
            {
                "time": per_step_wall(candidate) / per_step_wall(baseline),
                "gpu": candidate["memory"]["steady"]["gpu_peak_bytes"] / baseline["memory"]["steady"]["gpu_peak_bytes"],
                "host": candidate["memory"]["steady"]["host_peak_bytes"] / baseline["memory"]["steady"]["host_peak_bytes"],
            }
        )
    medians = {key: statistics.median(row[key] for row in ratios) for key in ("time", "gpu", "host")}
    benefit = (
        medians["time"] <= 1 / 1.10
        and all(row["time"] < 1 for row in ratios)
        or medians["gpu"] <= 0.90
        and all(row["gpu"] < 1 for row in ratios)
    )
    accepted = benefit and all(value <= 1.05 for value in medians.values())
    write_json(
        args.output,
        {
            "status": "PERFORMANCE_PASS" if accepted else "PERFORMANCE_REJECTED",
            "passed": accepted,
            "requires_separate_numerical_acceptance": True,
            "median_ratios": medians,
            "pair_ratios": ratios,
            "reports": bindings,
        },
    )
    if not accepted:
        raise SystemExit(1)


def add_workload(parser):
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--mul", type=int, default=32)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--outputs", choices=("single", "all"), default="single")
    parser.add_argument("--weights", choices=("none", "shared", "unshared"), default="shared")
    parser.add_argument("--alias-inputs", action="store_true")
    parser.add_argument("--workload", choices=("inference", "backward", "double-backward"), required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mode", choices=("measure", "profile"), default="measure")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--tensor-output", type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot", help="Run on the submission host before freezing sources")
    snap.add_argument("--source-root", type=Path, required=True)
    snap.add_argument("--revision-root", type=Path, help="Originating T-e3nn Git checkout for an immutable source copy")
    snap.add_argument("--output", type=Path, required=True)
    snap.set_defaults(func=snapshot)
    measure = commands.add_parser("run")
    add_workload(measure)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--sample-interval", type=float, default=0.2)
    measure.add_argument("--timeout", type=float, default=3600)
    measure.set_defaults(func=run)
    worker = commands.add_parser("_child", help=argparse.SUPPRESS)
    add_workload(worker)
    worker.add_argument("--control", type=Path, required=True)
    worker.set_defaults(func=child)
    tensors = commands.add_parser("compare-tensors")
    tensors.add_argument("--baseline", type=Path, required=True)
    tensors.add_argument("--candidate", type=Path, required=True)
    tensors.add_argument("--output", type=Path, required=True)
    tensors.set_defaults(func=compare_tensors)
    pairs = commands.add_parser("compare-pairs")
    pairs.add_argument("--pairs", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True)
    pairs.set_defaults(func=compare_pairs)
    args = parser.parse_args()
    if args.command in ("run", "_child"):
        if min(args.batch, args.mul, args.warmup, args.blocks, args.steps, args.profile_steps) < 1 or args.ell < 0:
            parser.error("Shapes and repetition counts must be positive")
        if args.family.startswith("tri-") and args.mul < 2:
            parser.error("Triangular workloads need mul >= 2; use uvuv for mul=1 controls")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
