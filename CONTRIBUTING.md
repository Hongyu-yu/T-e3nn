# Contribute

For the docstrings we use the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).

You can install some of the commonly used development tools by using e3nn's 'dev' extra:

```
pip install -e '.[dev]'
```

To have atomic code style checks performed at each commit, you can install the pre-commit hook using:

```
pre-commit install
```

These checks are automatically run on any commit made to the github repository but the pre-commit hook allows you to see if there are any problems locally.

Additionally, you may want to run the tests locally before pushing to remote.  This can be done with (from the root e3nn directory):

```
pytest tests
```

Tests must preserve the process-wide default dtype. For double-precision
gradcheck inputs, pass `dtype=torch.float64` to the tensor constructor instead
of calling `torch.set_default_dtype`. Tests that specifically exercise the
global default must restore it in `finally`. The session's `float_tolerance`
fixture sets a dtype only when its parameter changes; one leaking test can
otherwise make later tests labeled float32 silently run in float64.

On a shared compute cluster, run numerical tests, compilation and GPU
measurements inside an allocated job. Keep source snapshots and measurement
drivers immutable while those jobs use them.

The release CI lane uses CPython 3.10.20, PyTorch 2.11.0 CPU and the pinned
dependencies in `.github/requirements-ci.txt`. Install the matching scatter
and cluster wheels from the explicit PyG CPU index using
`.github/requirements-pyg-cpu.txt`; do not mix extension wheels from another
Torch/CUDA ABI. This lane replaces the historical Python 3.7/3.10 and Torch
1.11/1.12 matrix to align CI with the reviewed dependency versions. It does
not establish that all older versions allowed by package metadata have been
revalidated. The complete recorded T7 suite used CUDA-build Torch with CUDA
disabled; the new CPU-wheel lane must pass independently on GitHub.

The repository's `local_only` Actions policy is retained. Test CI uses reviewed
native `run` steps rather than external `uses` actions: it checks out the exact
public `GITHUB_SHA` with the runner's GitHub CLI, then verifies SHA-256 digests
before extracting pinned CPython 3.10.20 (20260807) and uv 0.9.22 under
`RUNNER_TEMP`. The job token is scoped only to the checkout step. Do not replace
this with an unpinned installer or relax the repository action policy.

The complete library test selection is `tests e3nn
benchmarks/test_tensor_product.py --doctest-modules`. Preserve its test cases,
dtype coverage, existing expected failures and numerical tolerances. Seed
Python, NumPy and Torch before collection, as the workflow does. The four
existing example doctests run in a separate process with
`--confcutdir=examples --rootdir=examples`: collecting the entire examples
tree imports executable data generators and collides with the library's
`conftest.py`. This separation retains their assertions without starting
dataset-generation commands. Benchmark contracts are tests; the measurement
CLI remains protected by its `__main__` guard and requires a Slurm allocation
for actual GPU work.

Fork test jobs do not require upstream upload secrets. PyPI publishing and
Coveralls upload are restricted to `e3nn/e3nn`; GitHub releases of T-e3nn
retain the `0.5.0+spin` package compatibility version.
JUnit counts are written to `GITHUB_STEP_SUMMARY` and stdout; complete pytest
output remains in the workflow logs for archival. No external artifact-upload
action is required. A platform startup failure with no jobs is not a test pass.

For formatting we use the [black](https://black.readthedocs.io/en/stable/index.html) library.
It can be installed with:

```
pip install black
```

and run with:

```
black .
```
