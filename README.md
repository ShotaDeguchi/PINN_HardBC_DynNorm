# PINN_HardBC_DynNorm

This is a TensorFlow implementation of the following paper:

Shota Deguchi, Mitsuteru Asai: Reliable and efficient inverse analysis using physics-informed neural networks with distance functions and adaptive weight tuning, JOURNAL NAME, 20XX (doi: XXXX)

## Environment
The code has been tested with the following dependencies:
- Python 3.6.8
- matplotlib 3.3.4
- numpy 1.19.5
- pyyaml 6.0
- scipy 1.5.4
- tensorflow 2.5.0

## Citation
If you use this code, please cite our paper as follows:

```bibtex
@article{Deguchi20XX,
  title={Reliable and efficient inverse analysis using physics-informed neural networks with distance functions and adaptive weight tuning},
  author={Shota Deguchi and Mitsuteru Asai},
  journal={JOURNAL NAME},
  year={20XX},
  doi={XXXX}
}
```

DynNorm stands for [(bias-corrected) Dynamic Normalization](https://github.com/ShotaDeguchi/DN_PINN), described in our paper:

```bibtex
@article{Deguchi_2023,
  doi = {10.1088/2399-6528/ace416},
  url = {https://dx.doi.org/10.1088/2399-6528/ace416},
  year = {2023},
  month = {jul},
  publisher = {IOP Publishing},
  volume = {7},
  number = {7},
  pages = {075005},
  author = {Shota Deguchi and Mitsuteru Asai},
  title = {Dynamic \& norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks},
  journal = {Journal of Physics Communications}
}
```

## License
MIT License
