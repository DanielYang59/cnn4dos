# MPhil Research Project at QUT: Convolutional Neural Networks and Volcano Plots: Screening and Prediction of Two-Dimensional Single-Atom Catalysts

[![arXiv](https://img.shields.io/badge/arXiv-2402.03876-b31b1b.svg)](https://arxiv.org/abs/2402.03876) [![MPhil Thesis](https://img.shields.io/badge/MPhil_Thesis-241290-b31b1b.svg)](https://eprints.qut.edu.au/241290/)

> [!NOTE]
> The development of the general purpose Python package has been relocated to:\
> [cat-scaling](https://github.com/DanielYang59/cat-scaling): Build Scaling Relations for Catalysts

[![arXiv](https://img.shields.io/badge/arXiv-2402.03876-b31b1b.svg)](https://arxiv.org/abs/2402.03876) [![MPhil Thesis](https://img.shields.io/badge/MPhil_Thesis-241290-b31b1b.svg)](https://eprints.qut.edu.au/241290/)

## Overview

This repository contains the code, data and resources to recreate my [MPhil research project](https://eprints.qut.edu.au/241290/) at Queensland University of Technology (QUT), titled "*Descriptor-guided screening and understanding of 2D substrate supported single atomic CO<sub>2</sub>RR electrocalysts*" and supervised by [Prof. Ting Liao](https://www.qut.edu.au/about/our-people/academic-profiles/t3.liao) and Prof. Ziqi Sun. The project attempted to propose a combined catalyst screening pipeline by predicting & understanding adsorption energy with density of states (DOS) via convolutional neural networks (CNN), and analysing adsorption energy with volcano plots.

## Description

This project investigated the CO<sub>2</sub> reduction to CH<sub>4</sub> reaction performance of 228 single atom electrocatalysts supported on six 2D substrates (graphitic carbon nitride (g-C<sub>3</sub>N<sub>4</sub>), nitrogen-doped graphene, graphene with dual-vacancy, black phosphorus, single-layer C<sub>2</sub>N and boron nitride). Furthermore, electronic descriptors and elementary descriptors were selected to elucidate the correlation between the intrinsic properties and catalytic performance of investigated candidates. Among them, electronic density of states (eDOS), inspired by the d-band theory, were selected to establish a direct descriptor-performance mapping through convolutional neural networks (CNNs). Resulted neural network model achieved mean absolute errors (MAEs) on the order of 0.1 eV for all nine intermediates (including the competing hydrogen evolution reaction (HER)).

## Highlights

- High prediction accuracy (MAE around 0.1 eV) for various intermediates, and free of intermediate-specific parameters.
- Minimal input data demand (only eDOS from supported single metal atom and adsorbate, not the entire catalyst).
- Enhanced model interpretability with occlusion experiments and shifting experiments.
- Physical meaningfulness confirmed by crystal orbital Hamilton population (COHP) analysis.
- Shifting experiments could potentially couple with volcano plots to improve existing candidates.

## Citation

If you find this work beneficial, kindly consider citing the [arXiv preprint](https://arxiv.org/abs/2402.03876) or [MPhil Thesis](https://eprints.qut.edu.au/241290/) (and a Journal Paper is in progress with my supervisory team).

```
@misc{yang2024convolutional,
      title={Convolutional Neural Networks and Volcano Plots: Screening and Prediction of Two-Dimensional Single-Atom Catalysts},
      author={Haoyu Yang and Juanli Zhao and Qiankun Wang and Bin Liu and Wei Luo and Ziqi Sun and Ting Liao},
      year={2024},
      eprint={2402.03876},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}

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

If you have any questions regarding this project, feel free to open a discussion or raise an issue.

I have provided as much resource data (VASP structures, eDOSs) as possible in this repo. However if you need access to the original dataset including VASP input/output files, please reach out to my supervisor [Prof. Ting Liao](https://www.qut.edu.au/about/our-people/academic-profiles/t3.liao) at QUT, as the original data is too large (in Terabytes) and has been stored/archived in the QUT Remote Data Storage System (RDSS) as per [QUT policy](https://airs.library.qut.edu.au/modules/8/3/), to which I no longer have access after graduation.

## References

- [My MPhil Thesis](https://eprints.qut.edu.au/241290/)
- [The arXiv Preprint](https://arxiv.org/abs/2402.03876)
- [Victor\'s Inspiring work on bimetallic surfaces](https://www.nature.com/articles/s41467-020-20342-6)
