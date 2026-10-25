# Optional LunaEquivariance adapter

`e3nn.o3.LunaTensorProduct` wraps an already-built T-e3nn
`o3.TensorProduct` and dispatches the contraction to LunaEquivariance when the
optional package is installed. The adapter preserves the original parameter
and buffer names, so existing checkpoints remain compatible.

The adapter validates the fork's time-reversal labels before dispatch. Luna's
native Irreps parser currently represents `(l, parity)` only; do not rebuild a
SpinGNN `(l, parity, time-reversal)` tensor product from Luna strings. Passing
an existing T-e3nn tensor product is required for spin/SOC experiments.

This is an opt-in eager/`torch.compile` backend. The default e3nn path is
unchanged when Luna is absent. Luna custom operators are not automatically
available to TorchScript or the SpinGNN LAMMPS PT2 loader. A deployed artifact
requires a separately registered C++/CUDA operator ABI and must pass the full
energy/force/spin-force/stress/virial parity matrix before production use.

The optional package currently requires e3nn 0.6 metadata, while the SpinGNN
fork identifies itself as `0.5.0+spin`. Install it only through the reviewed
compatibility environment and never let a resolver replace the SpinGNN fork.
