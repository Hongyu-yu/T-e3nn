# Time-reversal Euclidean neural networks

[![DOI](https://zenodo.org/badge/doi/10.48550/arXiv.2211.11403.svg)](https://doi.org/10.48550/arXiv.2211.11403)

T-e3nn is an extension of [e3nn](https://github.com/e3nn/e3nn) with consideration of time-reversal symmetry include quantities such as spin and velocity.

The aim of this library is to help the development of [Time-reversal](https://en.wikipedia.org/wiki/T-symmetry) [E(3)](https://en.wikipedia.org/wiki/Euclidean_group) equivariant neural networks.

It's built on [this version of e3nn](https://github.com/e3nn/e3nn/commit/b521bfcfcf4225ed500c15ec3419a24656f763ca)
 with nearly the same usage API considering Time-reversal and E(3) symmetry. So you can transfer your E(3) equivariant model into a Time-reversal E(3) equivariant model easily by initializing input considering time-reversal related irreps with T-e3nn. See more details in this [preprint](https://www.researchgate.net/publication/365607322_Time-reversal_equivariant_neural_network_potential_and_Hamiltonian_for_magnetic_materials).

## Installation
```
$ git clone https://github.com/Hongyu-yu/T-e3nn.git
$ cd T-e3nn/
$ pip install .
```
*Warning*: with T-e3nn installed, e3nn packages will be removed in your python environment while original code using e3nn should work fine as before. If any code based on e3nn originally works fine but not for T-e3nn which is carefully prevented during development, please submit an issue. Codes about `import e3nn` will be directed to T-e3nn instead. Please check the small difference of API listed below. Generally, very few changes including the initialization of the input irreps are needed to be made to transfer your model from e3nn to T-e3nn.

## Difference with E3NN

With a few changes on your original codes based on e3nn, time-reversal can be considered.

### **Irreps**:

#### Initialization

Usually, the only difference between e3nn and T-e3nn for network developer is to initialize the input of network considering time-reversal order.

In T-e3nn, `Irreps` are stored with `(l, p, t)` with `t` about time-reversal symmetry and `l` `p` from e3nn. 

You can initial `Irreps` by
- `Irrep(l, p, t)` or `Irrep(l, p)` with default `t=1`.
- `Irrep("lee")` or `Irrep("1e")` with default `t=1`
If you want to generate `Irrep` with odd time-reversal, you should include `t` when initializing `Irrep`.

Example:
- Irrep of spin vector should be `Irrep(1, 1, -1)` or `Irrep("1eo")`
- Irrep of bond vector can be `Irrep(1, -1)` or `Irrep("1o")` as the same in e3nn or `Irrep(1, -1, 1)` or `Irrep("1oe")` with explicit time-reversal index.
Iteration like `for mul, (l, p) in irreps` in e3nn should be modified as `for mul, (l, p, t) in irrep` 


#### **Property**

While `p` is about parity, `t` is about time-reversal. Here we use `T` as the time-reversal operation and `x` as the variable with `(l,p,t)`

For `x` of `t=1`, `Tx=x`. For most of physical quantities, `t=1`

For `x` of `t=-1`, `Tx=-x`. For the physical quantities related with time, such as velocity `v=dx/dt` and spin, `t=-1`.

`t` will be considered in the operation just like `p`.

When `t` of all variables is 1, it's degenerate into E(3) and act exactly the same as E3NN.

#### Class method difference
Difference below is barely used though.

Difference are highlighted with **bold**.
- D_from_angles(alpha, beta, gamma, k, **kt**=None)
- D_from_quaternion(q, k, **kt**)
- D_from_matrix(R, **parity**=True, **time_reversal**=False)
- spherical_harmonics(lmax, p=-1, **t**=1)
- **sort_array**
  - Sort the representations and return also the array index based on sort

### **Other API difference**
API below help to initialize the input and its irreps and test of the network.

- io.SphericalTensor(lmax, p_val, p_arg, **t_val**=1, **t_arg**=1)
- o3.SphericalHarmonics(..., **parity**=True, **time_reversal**=False)
- o3.spherical_harmonics(..., **parity**=True, **time_reversal**=False)
- util.test.assert_equivariant(...,**do_time_reversal**=True, **do_only_rot_spin**=False)
  - Whether to check time-reversal symmetry and whether spin-orbit effect existence.
  - If you want to check that SOC is turn off in your model, **do_only_rot_spin** should be true.
- util.test.equivariance_error(..., **do_time_reversal**=True, **do_only_rot_spin**=False)
- util._argtools._transform(..., **parity**=1, **tr_k**=0, **only_rot_spin**=False)
  - When checking model equivariance, just as `ireeps` of positions `r` represented with keyword "cartesian_points", all `"1eo"` vectors should be reprented with keyword "spin" such as spin, spin_force, velocity.

### Citing
If you use this repository in your work, please considering citing the preprint below and e3nn.
```
@misc{tenn_paper,
    doi = {10.48550/ARXIV.2211.11403},
    url = {https://arxiv.org/abs/2211.11403},
    author = {Hongyu Yu, Yang Zhong, Junyi Ji, Xingao Gong, Hongjun Xiang},
    keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences}, 
    title = {Time-reversal equivariant neural network potential and Hamiltonian for magnetic materials},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

README.md of E3nn:

# Euclidean neural networks
[![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn/badge.svg?branch=main)](https://coveralls.io/github/e3nn/e3nn?branch=main)
[![DOI](https://zenodo.org/badge/237431920.svg)](https://zenodo.org/badge/latestdoi/237431920)

**[Documentation](https://docs.e3nn.org)** | **[Code](https://github.com/e3nn/e3nn)** | **[ChangeLog](https://github.com/e3nn/e3nn/blob/main/ChangeLog.md)** | **[Colab](https://colab.research.google.com/drive/1Gps7mMOmzLe3Rt_b012xsz4UyuexTKAf?usp=sharing)**

The aim of this library is to help the development of [E(3)](https://en.wikipedia.org/wiki/Euclidean_group) equivariant neural networks.
It contains fundamental mathematical operations such as [tensor products](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html) and [spherical harmonics](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html).

![](https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif)

## Installation

**Important:** install pytorch and only then run the command

```
pip install --upgrade pip
pip install --upgrade e3nn
```

For details and optional dependencies, see [INSTALL.md](https://github.com/e3nn/e3nn/blob/main/INSTALL.md)

### Breaking changes
e3nn is under development.
It is recommanded to install using pip. The main branch is considered as unstable.
The second version number is incremented every time a breaking change is made to the code.
```
0.(increment when backwards incompatible release).(increment for backwards compatible release)
```

## Help
We are happy to help! The best way to get help on `e3nn` is to submit a [Question](https://github.com/e3nn/e3nn/issues/new?assignees=&labels=question&template=question.md&title=%E2%9D%93+%5BQUESTION%5D) or [Bug Report](https://github.com/e3nn/e3nn/issues/new?assignees=&labels=bug&template=bug-report.md&title=%F0%9F%90%9B+%5BBUG%5D).

## Want to get involved? Great!
If you want to get involved in and contribute to the development, improvement, and application of `e3nn`, introduce yourself in the [discussions](https://github.com/e3nn/e3nn/discussions/new).

## Code of conduct
Our community abides by the [Contributor Covenant Code of Conduct](https://github.com/e3nn/e3nn/blob/main/code_of_conduct.md).

## Citing
```
@misc{e3nn_paper,
    doi = {10.48550/ARXIV.2207.09453},
    url = {https://arxiv.org/abs/2207.09453},
    author = {Geiger, Mario and Smidt, Tess},
    keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences}, 
    title = {e3nn: Euclidean Neural Networks},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}

@software{e3nn,
  author       = {Mario Geiger and
                  Tess Smidt and
                  Alby M. and
                  Benjamin Kurt Miller and
                  Wouter Boomsma and
                  Bradley Dice and
                  Kostiantyn Lapchevskyi and
                  Maurice Weiler and
                  Michał Tyszkiewicz and
                  Simon Batzner and
                  Dylan Madisetti and
                  Martin Uhrin and
                  Jes Frellsen and
                  Nuri Jung and
                  Sophia Sanborn and
                  Mingjian Wen and
                  Josh Rackers and
                  Marcel Rød and
                  Michael Bailey},
  title        = {Euclidean neural networks: e3nn},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {0.5.0},
  doi          = {10.5281/zenodo.6459381},
  url          = {https://doi.org/10.5281/zenodo.6459381}
}
```

### Copyright

Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy),
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin
and Kostiantyn Lapchevskyi. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
