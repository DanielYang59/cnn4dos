# MPhil Research Project at QUT: Adsorption Energy Prediction from Electronic Density of States with Convolutional Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2402.03876-b31b1b.svg)](https://arxiv.org/abs/2402.03876) [![MPhil Thesis](https://img.shields.io/badge/MPhil_Thesis-241290-b31b1b.svg)](https://eprints.qut.edu.au/241290/)

## Overview

This repository contains the code, data and resources to support my [MPhil research project](https://eprints.qut.edu.au/241290/) at Queensland University of Technology (QUT), titled "*Descriptor-guided screening and understanding of 2D substrate supported single atomic CO<sub>2</sub>RR electrocalysts*" and co-supervised by [Prof. Ting Liao](https://www.qut.edu.au/about/our-people/academic-profiles/t3.liao) and [Prof. Ziqi Sun](https://www.qut.edu.au/about/our-people/academic-profiles/ziqi.sun). The project attempted to propose a combined catalyst screening pipeline by predicting & understanding adsorption energy with density of states (DOS) via convolutional neural networks (CNN), and analysing adsorption energy with volcano plots.

## Description

This project investigated the CO<sub>2</sub> reduction to CH<sub>4</sub> reaction performance of 228 single atom electrocatalysts supported on six 2D substrates (graphitic carbon nitride (g-C<sub>3</sub>N<sub>4</sub>), nitrogen-doped graphene, graphene with dual-vacancy, black phosphorus, single-layer C<sub>2</sub>N and boron nitride). Furthermore, electronic descriptors and elementary descriptors were selected to elucidate the correlation between the intrinsic properties and catalytic performance of investigated candidates. Among them, electronic density of states (eDOS), inspired by the d-band theory, were selected to establish a direct descriptor-performance mapping through convolutional neural networks (CNNs). Resulted neural network model achieved mean absolute errors (MAEs) on the order of 0.1 eV for all nine intermediates (including the competing hydrogen evolution reaction (HER)).

## Highlights

- High prediction accuracy (MAE around 0.1 eV) for various intermediates, and free of intermediate-specific parameters.
- Minimal input data demand (only eDOS from supported single metal atom and adsorbate, not the entire catalyst).
- Enhanced model interpretability with occlusion experiments and shifting experiments.
- Physical meaningfulness confirmed by crystal orbital Hamilton population (COHP) analysis.
- Shifting experiments could potentially couple with volcano plots to improve existing candidates.

## TODOs

- Cleaning up the code into general and reusable modules.

## Citation

If you find this work beneficial, kindly consider citing [my MPhil Thesis](https://eprints.qut.edu.au/241290/) (and a Journal Paper is in progress with my supervisory team).

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
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contact and Data Access

If you have **ANY** questions regarding this project, feel free to open a discussion, raise an issue, or contact me directly at [yanghaoyu97@outlook.com](yanghaoyu97@outlook.com).

I have provided as much resource data (VASP structures, eDOSs) as possible in this repo. However if you need access to the original VASP input/output files, please reach out to my supervisor [Prof. Ting Liao](https://www.qut.edu.au/about/our-people/academic-profiles/t3.liao) in QUT, as the original data is too large (in several TBs) and has been stored in the QUT Remote Data Storage System (RDSS), to which I no longer have access after graduation.

## References

- [My MPhil Thesis](https://eprints.qut.edu.au/241290/)
- [Victor\'s Inspiring work on bimetallic surfaces](https://www.nature.com/articles/s41467-020-20342-6)
