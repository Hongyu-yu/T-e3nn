"""Regression contracts for bounded operator measurements and strict evidence."""

import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import types

import pytest


spec = importlib.util.spec_from_file_location("tp_benchmark", Path(__file__).with_name("tensor_product.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
BARE_UUID = "f142680b-8e8e-6e5c-5579-255a012bb587"
GPU_UUID = "GPU-" + BARE_UUID
OTHER_UUID = "GPU-f142680b-8e8e-6e5c-5579-255a012bb588"


def test_atomic_report_refuses_overwrite_and_nonfinite(tmp_path):
    path = tmp_path / "report.json"
    bench.write_json(path, {"status": "RECORDED"})
    with pytest.raises(FileExistsError):
        bench.write_json(path, {"status": "PASS"})
    assert json.loads(path.read_text()) == {"status": "RECORDED"}
    with pytest.raises(ValueError):
        bench.write_json(tmp_path / "bad.json", {"time": float("nan")})


def test_atomic_report_cleanup_supports_legacy_unlink(tmp_path, monkeypatch):
    original_unlink = Path.unlink

    def legacy_unlink(path):
        return original_unlink(path)

    monkeypatch.setattr(Path, "unlink", legacy_unlink)
    path = tmp_path / "report.json"
    bench.write_json(path, {"phase": "cold"}, replace=True)
    bench.write_json(path, {"phase": "steady"}, replace=True)
    assert json.loads(path.read_text()) == {"phase": "steady"}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_phase_write_rejects_nonfinite_without_losing_previous_state(tmp_path, bad_value):
    path = tmp_path / "phase.json"
    bench.write_json(path, {"phase": "steady"}, replace=True)
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        bench.write_json(path, {"phase": "complete", "memory": bad_value}, replace=True)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("failure", ["fsync", "publish"])
def test_atomic_write_failure_preserves_destination_and_cleans_temporary(tmp_path, monkeypatch, replace, failure):
    path = tmp_path / "phase.json"
    previous = b'{"phase": "cold"}\n' if replace else None
    if previous is not None:
        path.write_bytes(previous)

    def fail(*args):
        raise OSError("injected persistence failure")

    operation = "fsync" if failure == "fsync" else "replace" if replace else "link"
    monkeypatch.setattr(bench.os, operation, fail)
    with pytest.raises(OSError, match="injected persistence failure"):
        bench.write_json(path, {"phase": "steady"}, replace=replace)
    if replace:
        assert path.read_bytes() == previous
        assert list(tmp_path.iterdir()) == [path]
    else:
        assert not path.exists()
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("replace", [False, True])
def test_atomic_write_syncs_before_publishing(tmp_path, monkeypatch, replace):
    path = tmp_path / "phase.json"
    fsync = bench.os.fsync
    operation = "replace" if replace else "link"
    publish = getattr(bench.os, operation)
    events = []

    def sync(fd):
        events.append("fsync")
        return fsync(fd)

    def publish_after_sync(source, destination):
        assert events == ["fsync"]
        events.append("publish")
        return publish(source, destination)

    monkeypatch.setattr(bench.os, "fsync", sync)
    monkeypatch.setattr(bench.os, operation, publish_after_sync)
    bench.write_json(path, {"phase": "steady"}, replace=replace)
    assert events == ["fsync", "publish"]
    assert json.loads(path.read_text()) == {"phase": "steady"}


def test_phase_writes_do_not_accumulate_encoder_cycles_with_gc_disabled(tmp_path):
    def encoder_functions():
        return {
            id(value)
            for value in gc.get_objects()
            if isinstance(value, types.FunctionType) and value.__module__ == "json.encoder"
        }

    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        before = encoder_functions()
        collections = [generation["collections"] for generation in gc.get_stats()]
        for block in range(128):
            phase = "block-{}".format(block)
            bench.write_json(tmp_path / "phase.json", {"phase": phase, "uuid": GPU_UUID}, replace=True)
            bench.write_json(tmp_path / "ack.json", {"phase": phase}, replace=True)
            bench.write_json(
                tmp_path / "checkpoint.json",
                {"phase": phase, "completed_steps": block * 100, "allocator": {"allocated_bytes": 4096}},
                replace=True,
            )
            if block in (63, 127):
                retained = encoder_functions() - before
                assert not retained, "Repeated status writes retained {} encoder functions".format(len(retained))
        assert [generation["collections"] for generation in gc.get_stats()] == collections
        # Positive control: the observer must detect the old recursive encoder's
        # unreachable cycles while collection stays disabled.
        with (tmp_path / "pretty-control.json").open("w") as handle:
            json.dump({"phase": "control", "allocator": {"allocated_bytes": 4096}}, handle, indent=2)
        assert encoder_functions() > before
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def test_import_guard_supports_legacy_path_and_rejects_siblings(tmp_path, monkeypatch):
    def unsupported_relative_check(*args):
        raise AssertionError("is_relative_to is unavailable on supported older Python")

    monkeypatch.setattr(Path, "is_relative_to", unsupported_relative_check, raising=False)
    package = tmp_path / "e3nn"
    modules = {"e3nn": argparse.Namespace(__file__=str(package / "__init__.py"))}
    monkeypatch.setattr(bench, "sys", argparse.Namespace(modules=modules))
    assert bench.assert_imports(tmp_path) == {"e3nn": str(package / "__init__.py")}
    modules["e3nn.foreign"] = argparse.Namespace(__file__=str(tmp_path / "e3nn-other" / "foreign.py"))
    with pytest.raises(ValueError, match="Foreign e3nn import"):
        bench.assert_imports(tmp_path)


def test_source_guard_rejects_nonanchor_mutation(tmp_path):
    package = tmp_path / "e3nn"
    package.mkdir()
    for name in ["__init__"] + ["source_{}".format(i) for i in range(20)]:
        (package / (name + ".py")).write_text("value = 1\n")
    files = bench.source_files(tmp_path)
    manifest = tmp_path / "manifest.json"
    bench.write_json(
        manifest,
        {
            "schema": 1,
            "root": str(tmp_path),
            "head": "a" * 40,
            "files": files,
            "tree_sha256": bench.digest_json(files),
            "driver_sha256": bench.digest_file(bench.__file__),
        },
    )
    bench.validate_source(tmp_path, manifest)
    (package / "source_19.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="manifest mismatch"):
        bench.validate_source(tmp_path, manifest)


@pytest.mark.parametrize("problem", ["wrong_uuid", "missing", "extra_context", "duplicate", "pid_reused"])
def test_process_sampling_fails_closed(monkeypatch, problem):
    identity = {"pid": 7, "start_ticks": 100}
    monkeypatch.setattr(
        bench, "process_identity", lambda pid: {**identity, "start_ticks": 101} if problem == "pid_reused" else identity
    )
    monkeypatch.setattr(bench, "host_sample", lambda pid: {"VmRSS": 12, "VmHWM": 15})
    row = {"pid": 7, "uuid": GPU_UUID, "bytes": 1024}
    rows = [] if problem == "missing" else [{**row, "uuid": OTHER_UUID}] if problem == "wrong_uuid" else [row]
    if problem == "extra_context":
        rows.append({**row, "uuid": OTHER_UUID})
    if problem == "duplicate":
        rows.append(dict(row))
    monkeypatch.setattr(bench, "gpu_rows", lambda: rows)
    with pytest.raises(ValueError):
        bench.sample_process(identity, GPU_UUID)


@pytest.mark.parametrize("representation", [BARE_UUID, GPU_UUID, BARE_UUID.upper(), "GPU-" + BARE_UUID.upper()])
def test_real_torch_nvml_uuid_formats_match(monkeypatch, representation):
    identity = {"pid": 7, "start_ticks": 100}
    raw_line = "7, {}, 512".format(GPU_UUID)
    monkeypatch.setattr(bench, "process_identity", lambda pid: identity)
    monkeypatch.setattr(bench, "host_sample", lambda pid: {"VmRSS": 12, "VmHWM": 15})
    monkeypatch.setattr(bench.subprocess, "run", lambda *args, **kwargs: argparse.Namespace(stdout=raw_line + "\n"))
    sample = bench.sample_process(identity, representation)
    assert sample["gpu_uuid"] == GPU_UUID
    assert sample["gpu_bytes"] == 512 * 1024 * 1024
    assert sample["gpu_row"]["raw_line"] == raw_line
    assert sample["gpu_row"]["uuid_raw"] == GPU_UUID


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        0,
        "GPU-A",
        "GPU-GPU-" + BARE_UUID,
        "MIG-" + BARE_UUID,
        BARE_UUID.replace("-", ""),
        "{" + BARE_UUID + "}",
        BARE_UUID + " ",
        "gpu-" + BARE_UUID,
    ],
)
def test_uuid_normalization_rejects_other_identifiers(invalid):
    with pytest.raises(ValueError, match="Invalid physical GPU UUID"):
        bench.normalize_gpu_uuid(invalid)


def test_wrong_uuid_retains_actual_nvml_row(monkeypatch):
    identity = {"pid": 7, "start_ticks": 100}
    raw_line = "7, {}, 512".format(OTHER_UUID)
    monkeypatch.setattr(bench, "process_identity", lambda pid: identity)
    monkeypatch.setattr(bench, "host_sample", lambda pid: {"VmRSS": 12, "VmHWM": 15})
    monkeypatch.setattr(bench.subprocess, "run", lambda *args, **kwargs: argparse.Namespace(stdout=raw_line + "\n"))
    with pytest.raises(bench.GPUProcessEvidenceError) as caught:
        bench.sample_process(identity, BARE_UUID)
    assert caught.value.diagnostics["observed_rows"][0]["raw_line"] == raw_line
    assert caught.value.diagnostics["expected_uuid"] == GPU_UUID


@pytest.mark.parametrize("family", bench.FAMILIES)
@pytest.mark.parametrize("workload", ["inference", "backward", "double-backward"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_real_workloads_reuse_inputs_and_release_graph(family, workload, dtype):
    import torch
    from e3nn import o3

    args = argparse.Namespace(
        family=family,
        mul=3,
        ell=2,
        outputs="all",
        weights="shared",
        alias_inputs=False,
        dtype=dtype,
        device="cpu",
        seed=1729,
        batch=2,
        workload=workload,
    )
    _, step, description = bench.make_workload(args, torch, o3)
    first, second = step(), step()
    assert all(value.grad_fn is None for value in second)
    assert all(torch.isfinite(value).all() for value in second)
    assert all(value.shape == repeated.shape and torch.equal(value, repeated) for value, repeated in zip(first, second))
    if workload != "inference":
        assert all(value.norm() > 0 for value in first[1:])
    _, other_step, other_description = bench.make_workload(copy.copy(args), torch, o3)
    assert description == other_description
    assert all(torch.equal(value, repeated) for value, repeated in zip(first, other_step()))


@pytest.mark.parametrize("weights", ["shared", "unshared"])
def test_alias_input_double_backward_and_weight_participation(weights):
    import torch
    from e3nn import o3

    args = argparse.Namespace(
        family="tri-weighted",
        mul=3,
        ell=1,
        outputs="all",
        weights=weights,
        alias_inputs=True,
        dtype="float64",
        device="cpu",
        seed=1729,
        batch=2,
        workload="double-backward",
    )
    _, step, description = bench.make_workload(args, torch, o3)
    values = step()
    assert len(values) == 4  # output, shared-input force, input loss gradient, weight loss gradient
    assert description["input_digests"]["input2"]["alias_of"] == "input1"
    assert all(torch.isfinite(value).all() and value.norm() > 0 for value in values)


def test_cpu_preflight_cannot_be_formal_acceptance():
    report = {
        "schema": 1,
        "status": "RECORDED",
        "passed": False,
        "run_complete": True,
        "errors": [],
        "returncode": 0,
        "mode": "measure",
        "workload": {},
        "workload_sha256": bench.digest_json({}),
        "source": {"files": {str(i): "a" * 64 for i in range(20)}, "driver_sha256": "b" * 64},
        "driver_sha256": "b" * 64,
        "numerical": {"finite": True},
        "runtime": {},
        "allocation": None,
    }
    report["source"]["tree_sha256"] = bench.digest_json(report["source"]["files"])
    with pytest.raises(ValueError, match="Slurm GPU"):
        bench.validate_report(report)


def paired_fixture(tmp_path, ratio=0.8):
    pairs = []
    serial = 0
    files = {"e3nn/__init__.py": "a" * 64, **{"e3nn/file{}.py".format(i): "a" * 64 for i in range(20)}}
    for index in range(5):
        order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        pair = {"order": order}
        for arm in order:
            serial += 1
            folder = tmp_path / "{}-{}".format(index, arm)
            folder.mkdir()
            samples = [
                {
                    "phase": phase,
                    "VmRSS": 100,
                    "VmHWM": 120,
                    "gpu_bytes": 1000,
                    "pid": serial,
                    "start_ticks": 1000 + serial,
                    "gpu_uuid": "GPU-A",
                }
                for phase in ("cold", "steady")
            ]
            bench.write_json(folder / "samples.json", samples)
            report = {
                "schema": 1,
                "status": "RECORDED",
                "passed": False,
                "run_complete": True,
                "errors": [],
                "returncode": 0,
                "mode": "measure",
                "workload": {},
                "workload_sha256": bench.digest_json({}),
                "source": {"files": files, "tree_sha256": bench.digest_json(files), "driver_sha256": "b" * 64},
                "driver_sha256": "b" * 64,
                "measurement": {"sample_interval_seconds": 0.2, "sampler": "independent_parent_process"},
                "numerical": {"finite": True},
                "runtime": {"gpu": {"uuid": "GPU-A"}},
                "allocation": "job",
                "hostname": "node",
                "warmup": 20,
                "process": {"pid": serial, "start_ticks": 1000 + serial},
                "pid": serial,
                "started_monotonic_ns": serial * 100,
                "finished_monotonic_ns": serial * 100 + 50,
                "blocks": [{"steps": 200, "wall_seconds": ratio if arm == "candidate" else 1.0} for _ in range(5)],
                "memory": {
                    "scope": "child_pid_and_uuid",
                    "window": "post_setup_through_measurement",
                    "samples_sha256": bench.digest_file(folder / "samples.json"),
                    **{phase: bench.summarize_samples(samples, phase) for phase in ("cold", "steady")},
                },
            }
            path = folder / "report.json"
            bench.write_json(path, report)
            pair[arm] = str(path)
        pairs.append(pair)
    path = tmp_path / "pairs.json"
    bench.write_json(path, {"pairs": pairs})
    return argparse.Namespace(pairs=path, output=tmp_path / "comparison.json"), pairs


def numerical_fixture(tmp_path, monkeypatch, baseline, candidate):
    import torch

    _, pairs = paired_fixture(tmp_path)
    paths = [Path(pairs[0][arm]) for arm in ("baseline", "candidate")]
    artifacts = {}
    for path, tensor in zip(paths, (baseline, candidate)):
        artifact = path.with_name("tensors.pt")
        artifact.write_bytes(b"Controlled tensor loader fixture")
        report = json.loads(path.read_text())
        report["numerical"]["artifact"] = {"path": str(artifact), "sha256": bench.digest_file(artifact)}
        path.write_text(json.dumps(report))
        artifacts[str(artifact)] = (tensor,)

    def load_fixture(path, map_location, weights_only):
        # Protocol tests remain collectable on Torch 1.11, without enabling unsafe loading.
        assert map_location == "cpu" and weights_only is True
        return artifacts[path]

    monkeypatch.setattr(torch, "load", load_fixture)
    return argparse.Namespace(baseline=paths[0], candidate=paths[1], output=tmp_path / "numerical.json")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numerical_comparison_records_explicit_tolerances(tmp_path, monkeypatch, dtype):
    import torch

    baseline = torch.tensor([0.0, 1.0], dtype=getattr(torch, dtype))
    args = numerical_fixture(tmp_path, monkeypatch, baseline, baseline.clone())
    bench.compare_tensors(args)
    report = json.loads(args.output.read_text())
    expected = {"rtol": 1.3e-6, "atol": 1e-5} if dtype == "float32" else {"rtol": 1e-9, "atol": 1e-10}
    assert report["tensor_tolerances"] == [{"dtype": "torch." + dtype, **expected}]
    assert report["tolerance"] == "explicit_dtype_tolerances_v1"
    assert report["comparator_sha256"] == bench.digest_file(bench.__file__)


@pytest.mark.parametrize("dtype,delta", [("float32", 2e-5), ("float64", 1e-8)])
@pytest.mark.parametrize("magnitude", [0.0, 1.0])
def test_numerical_comparison_rejects_relaxed_gate(tmp_path, monkeypatch, dtype, delta, magnitude):
    import torch

    baseline = torch.tensor([magnitude], dtype=getattr(torch, dtype))
    candidate = baseline + delta
    args = numerical_fixture(tmp_path, monkeypatch, baseline, candidate)
    with pytest.raises(AssertionError):
        bench.compare_tensors(args)
    assert not args.output.exists()


@pytest.mark.parametrize("candidate_dtype", ["float32", "int64"])
def test_numerical_comparison_rejects_dtype_mismatch(tmp_path, monkeypatch, candidate_dtype):
    import torch

    baseline = torch.ones(1, dtype=torch.float64)
    candidate = torch.ones(1, dtype=getattr(torch, candidate_dtype))
    args = numerical_fixture(tmp_path, monkeypatch, baseline, candidate)
    with pytest.raises(ValueError, match="matching float32 or float64"):
        bench.compare_tensors(args)


def test_formal_pairs_require_measured_benefit(tmp_path):
    args, _ = paired_fixture(tmp_path, ratio=1.0)
    with pytest.raises(SystemExit):
        bench.compare_pairs(args)
    assert json.loads(args.output.read_text())["status"] == "PERFORMANCE_REJECTED"


def test_formal_pairs_accept_only_performance(tmp_path):
    args, _ = paired_fixture(tmp_path)
    bench.compare_pairs(args)
    result = json.loads(args.output.read_text())
    assert result["status"] == "PERFORMANCE_PASS"
    assert result["requires_separate_numerical_acceptance"] is True


@pytest.mark.parametrize("problem", ["window", "uuid", "workload", "source", "samples", "warmup", "blocks", "pid"])
def test_formal_pairs_reject_invalid_evidence(tmp_path, problem):
    args, pairs = paired_fixture(tmp_path)
    path = Path(pairs[1]["candidate"])
    report = json.loads(path.read_text())
    if problem == "window":
        report["started_monotonic_ns"] = 1
    elif problem == "uuid":
        report["runtime"]["gpu"]["uuid"] = "GPU-B"
    elif problem == "workload":
        report["workload"] = {"batch": 1}
        report["workload_sha256"] = bench.digest_json(report["workload"])
    elif problem == "source":
        report["source"]["files"]["e3nn/file1.py"] = "c" * 64
        report["source"]["tree_sha256"] = bench.digest_json(report["source"]["files"])
    elif problem == "samples":
        (path.parent / "samples.json").write_text("[]\n")
    elif problem == "warmup":
        report["warmup"] = 19
    elif problem == "blocks":
        report["blocks"].pop()
    else:
        report["process"]["pid"] = 1
        report["pid"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        bench.compare_pairs(args)
