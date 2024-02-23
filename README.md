## Deep Learning with Electronic Density of States and Scaling Relations

[![arXiv](https://img.shields.io/badge/arXiv-2402.03876-b31b1b.svg)](https://arxiv.org/abs/2402.03876)
[![MPhil Thesis](https://img.shields.io/badge/MPhil_Thesis-241290-b31b1b.svg)](https://eprints.qut.edu.au/241290/)

> [!WARNING]
> The entire codebase is being refactored into a generally usable package,
> and frequent and breaking changes are to be expected.

> [!NOTE]
> This repo is being refactored into a generally usable package, and if you
> come to view the original code and data to recreate the results for
> my MPhil thesis or the Preprint paper, please move to the
> `arXiv-06Feb2024` branch (arXiv submission on archived 06 Feb 2024).

## Introduction

This project originates from my MPhil research project titled *Descriptor-guided screening and understanding of 2D substrate-supported single-atomic CO<sub>2</sub>RR electrocatalysts*. In this study, a convolutional neural network (CNN) was developed to predict adsorption energy from electronic density of states (eDOS).

## Manual

A manual (docs) site would be built once I generally finish refactoring the codebase.

## Citation

If you find this work beneficial, kindly consider citing [my MPhil Thesis](https://eprints.qut.edu.au/241290/) or the [preprint version](https://arxiv.org/abs/2402.03876) on arXiv (a Journal Paper is in progress with my supervisory team).

```
@phdthesis{quteprints241290,
          school = {Queensland University of Technology},
          author = {Haoyu Yang},
           title = {Descriptor-guided screening and understanding of 2D substrate supported single atomic CO2RR electrocatalysts},
           month = {July},
            year = {2023},
        abstract = {This thesis investigated the CO2 reduction reaction performance of 228 supported single atom catalysts via density functional theory method. Furthermore, electronic descriptors and elementary descriptors were selected trying to elucidate the correlation between catalytic performance and intrinsic performance of investigated candidates. Among them, electronic density of states, inspired by the d-band theory, were selected to establish a direct descriptor-performance mapping through neural networks. Resulted neural network model achieved a mean absolute error on the order of 0.1 eV.},
             url = {https://eprints.qut.edu.au/241290/},
             doi = {10.5204/thesis.eprints.241290},
        keywords = {Adsorption Energy, CO2 Reduction Reaction, Convolutional Neural Network, Density Functional Theory, Descriptor, Electronic Density of States, Linear Scaling Relation, Limiting Potential, Single-Atom Catalyst, Two-Dimensional}
}

@misc{yang2024convolutional,
      title={Convolutional Neural Networks and Volcano Plots: Screening and Prediction of Two-Dimensional Single-Atom Catalysts},
      author={Haoyu Yang and Juanli Zhao and Qiankun Wang and Bin Liu and Wei Luo and Ziqi Sun and Ting Liao},
      year={2024},
      eprint={2402.03876},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contact and Data Access

If you have ANY questions regarding this project, feel free to open a discussion, raise an issue, or contact me directly at [yanghaoyu97@outlook.com](yanghaoyu97@outlook.com).

I have provided as much resource (VASP structures, eDOS data in numpy format and more) as possible in this repo. However if you need access to the original VASP input/output files, please reach out to my supervisor [Prof. Ting Liao](https://www.qut.edu.au/about/our-people/academic-profiles/t3.liao) in QUT, as the original data is too large (in several TBs) and has been stored in the QUT Remote Data Storage System (RDSS), to which I no longer have access after graduation.

## References

- [My MPhil Thesis at QUT](https://eprints.qut.edu.au/241290/)
- [arXiv Paper](https://arxiv.org/abs/2402.03876)
