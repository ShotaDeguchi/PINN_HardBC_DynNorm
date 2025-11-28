# PINN_HardBC_DynNorm
This is a TensorFlow implementation of the following paper:

Deguchi, S. and Asai, M.: Reliable and efficient inverse analysis using physics-informed neural networks with normalized distance functions and adaptive weight tuning, *Machine Learning: Science and Technology*, Vol. 6, No. 4, pp. 045055, 2025 ([https://doi.org/10.1088/2632-2153/ae1b71](https://doi.org/10.1088/2632-2153/ae1b71), [arXiv: 2504.18091](https://arxiv.org/abs/2504.18091)).

## Overview
```
PINN_HardBC_DynNorm
├─01_poisson                # Forward problem: Poisson equation with mixed boundary conditions
│  ├─00_inhmgDir_hmgNeu     # inhomogeneous Dirichlet and homogeneous Neumann conditions
│  │  ├─00_soft
│  │  ├─01_hard
│  │  └─99_reference
│  └─01_inhmgDir_inhmgNeu   # inhomogeneous Dirichlet and inhomogeneous Neumann conditions
│      ├─00_soft
│      ├─01_hard
│      └─99_reference
└─02_cavity                 # Inverse problem: shear-driven cavity flow
    └─00_Re1000
        ├─00_soft_dynnorm
        ├─01_hard_dynnorm
        └─99_reference
```

## Environment
The code has been tested with the following dependencies:
- python 3.6.8
  - numpy 1.19.5
  - scipy 1.5.4
  - tensorflow 2.5.0
  - pyyaml 6.0.2

- python 3.10.16
  - numpy 1.26.4
  - scipy 1.15.1
  - tensorflow 2.16.1
  - pyyaml 6.0.2

## Citation
Please cite us as follows:
```bibtex
@article{Deguchi2025MLST,
	doi = {10.1088/2632-2153/ae1b71},
	url = {https://doi.org/10.1088/2632-2153/ae1b71},
	year = {2025},
	month = {nov},
	publisher = {IOP Publishing},
	volume = {6},
	number = {4},
	pages = {045055},
	author = {Deguchi, Shota and Asai, Mitsuteru},
	title = {Reliable and efficient inverse analysis using physics-informed neural networks with normalized distance functions and adaptive weight tuning},
	journal = {Machine Learning: Science and Technology}
}
```
```bibtex
@misc{Deguchi2025arXiv,
  title={Reliable and efficient inverse analysis using physics-informed neural networks with normalized distance functions and adaptive weight tuning},
  author={Shota Deguchi and Mitsuteru Asai},
  year={2025},
  eprint={2504.18091},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.18091},
}
```
DynNorm stands for [bias-corrected dynamic normalization](https://github.com/ShotaDeguchi/DN_PINN), described in the following paper:
```bibtex
@article{Deguchi2023DynNorm,
  doi={10.1088/2399-6528/ace416},
  url={https://dx.doi.org/10.1088/2399-6528/ace416},
  year={2023},
  month={jul},
  publisher={IOP Publishing},
  volume={7},
  number={7},
  pages={075005},
  author={Shota Deguchi and Mitsuteru Asai},
  title={Dynamic \& norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks},
  journal={Journal of Physics Communications}
}
```
The reference solution for the shear-driven cavity flow is obtained using the third-order upwind and second-order central finite difference method, implemented [here](https://github.com/ShotaDeguchi/Cavity_FDM_NumPy2).

## License
MIT License


