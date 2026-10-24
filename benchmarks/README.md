# Tensor-product measurements

Measurement execution requires Python 3.10 and the pinned PyTorch 2.11 environment.
The pure protocol helpers and their tests preserve the library's Python 3.7 and
older Torch CI compatibility; they do not relax the measurement runtime guard.
The library's Python compatibility is unchanged. All numerical execution,
profiling and tests run inside Slurm; snapshot creation uses only the standard
library and Git on the submission host. GPU submission belongs to the project
budget controller.

First freeze each arm's complete `e3nn/` tree. Create an individual manifest on
the submission host with `snapshot --source-root FROZEN_ROOT --revision-root
ORIGINATING_GIT_CHECKOUT --output MANIFEST`. It records the actual revision,
checkout dirty list, complete Python content map, source root and driver hash.
The frozen source map, rather than the revision alone, identifies execution.
Freeze the benchmark driver as well; do not edit a driver used by running jobs.

The `run` command launches one fresh child and records `RECORDED`, never `PASS`.
Specify `--source-root`, `--manifest`, `--family`, `--workload` and `--output`.
The default work is 20 warmup steps followed by five blocks of 200 steps.
`--family` selects `tri-unweighted`, `tri-weighted`, or `uvuv-scalar-left`,
`uvuv-scalar-right`, `uvuv-scalar-output`; `--workload` selects `inference`,
`backward`, or `double-backward`. The last mode differentiates a nonlinear
scalar energy with a seeded, nonconstant cotangent, then differentiates a
force loss through the input gradient and all weights. No GPU input lists or
graphs accumulate across iterations. `--alias-inputs` exercises square-style
shared input gradients; `--weights unshared` exercises per-sample weights.

Use `--batch 4096 --mul 32 --ell 2` for the large operator case, with smaller
128/1024 batch probes first. Include `--batch 1 --mul 2 --ell 0` triangular
controls and `--batch 1 --mul 1 --ell 2` uvuv controls. `--outputs all` includes
all allowed angular outputs of triangular products and exposes shared
intermediates. Run both float32 and float64 correctness, including alias inputs.

Source imports and complete source hashes are checked before and after work.
Inputs, weights and cotangents come from a private CPU generator, independent
of constructor RNG consumption. Reports record their byte digests, dtype,
determinism, TF32 policy, device UUID, runtime, process identity and actual
generated graph. Each timed block records synchronized host and CUDA-event
time. The parent samples the child's PID and UUID and its Linux RSS; wrong
UUIDs, extra GPU contexts, missing samples and PID reuse invalidate the run.
Physical GPU UUIDs accept only the standard hexadecimal UUID representation
with an optional `GPU-` prefix. Reports retain the canonical identifier, Torch's
original representation, and each original `nvidia-smi` row; normalization never
merges process rows or hides an additional context on another GPU.
Cold sampling begins after input/module setup and includes warmup. Steady
sampling includes the measurement boundary handshake. This is not full
process-lifetime memory. Allocated/reserved CUDA memory is reported separately.

Frequently replaced phase, acknowledgement and checkpoint files use compact
one-shot JSON serialization. CPython's streaming pretty encoder creates
recursive reference cycles, which can remain until garbage collection and
distort host-memory samples. One-time reports retain readable indentation.
Both paths preserve finite-value checks, temporary-file cleanup, fsync and
atomic publication; historical reports still cannot be overwritten. The
protocol never forces garbage collection during measurement, and its memory
growth gates are unchanged. A writer change requires a new frozen driver and
new source manifests; old measurements retain their original driver identity.

Use a separate run with `--mode profile` for a Chrome trace and operator table;
it is never eligible for formal timing. For numerical comparison, separate
short runs can use `--tensor-output FILE`; then `compare-tensors --baseline
REPORT --candidate REPORT --output CHECK` compares every output and participating
derivative using explicit dtype-specific tolerances: float32 retains
`rtol=1.3e-6, atol=1e-5`, while float64 uses `rtol=1e-9, atol=1e-10`. Reports
record the dtype and both tolerances for each tensor, together with the comparator
source hash. Mismatched or unsupported
dtypes are rejected. These gates do not depend on changing PyTorch defaults. This
operator probe complements the library's gradcheck, symmetry and export tests.

Run ten fresh processes in one allocation, in five alternating AB/BA pairs.
The input to `compare-pairs --pairs PAIRS_JSON --output RESULT` has the form
`{"pairs": [{"order": ["baseline", "candidate"], "baseline": ".../report.json",
"candidate": ".../report.json"}, ...]}`. The comparator checks actual execution
windows, source/runtime/workload consistency, process independence, and raw
memory records. It requires at least 10% median speed improvement or at least
10% median process-GPU-memory reduction with the same direction in every pair;
median time, GPU memory and host RSS regressions cannot exceed 5%.
`PERFORMANCE_PASS` still requires separate numerical and end-to-end acceptance;
operator gains are not claims about SpinGNN throughput.

An orchestration controller must stop the entire matrix after an invalid or
incomplete measurement, including a nonzero measurement exit. Repeating the
remaining workloads cannot repair process ownership or provenance failures.
Numerical comparisons that fail after valid measurements may continue to collect
the remaining correctness cases, while retaining their failed status.
