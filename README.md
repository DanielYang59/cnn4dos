# MPhil Research Project "cnn4dos": Adsorption Energy Prediction from Electronic Density of States with Convolutional Neural Networks

## Overview

This repository contains the code, data and resources to support my [MPhil research project](https://eprints.qut.edu.au/241290/) at Queensland University of Technology (QUT). The project attempted to propose a combined catalyst screening pipeline by predicting & understanding adsorption energy with density of states (DOS) via convolutional neural networks (CNN), and analysing adsorption energy with volcano plots.

## Description

This project investigated the CO<sub>2</sub> reduction to CH<sub>4</sub> reaction performance of 228 single atom electrocatalysts supported on six 2D substrates (graphitic carbon nitride (g-C<sub>3</sub>N<sub>4</sub>), nitrogen-doped graphene, graphene with dual-vacancy, black phosphorus, single-layer C<sub>2</sub>N and boron nitride). Furthermore, electronic descriptors and elementary descriptors were selected to elucidate the correlation between the intrinsic properties and catalytic performance of investigated candidates. Among them, electronic density of states (eDOS), inspired by the d-band theory, were selected to establish a direct descriptor-performance mapping through convolutional neural networks (CNNs). Resulted neural network model achieved mean absolute errors (MAEs) on the order of 0.1 eV for all nine intermediates (including the competing hydrogen evolution reaction (HER)).

## Highlights

- High prediction accuracy (MAE around 0.1 eV) for various intermediates, and free of intermediate-specific parameters.
- Minimal input data demand (only eDOS from supported single metal atom and adsorbate, not the entire catalyst).
- Enhanced model interpretability with occlusion experiments and shifting experiments.
- Physical meaningfulness confirmed by crystal orbital Hamilton population (COHP) analysis.
- Shifting experiments could potentially couple with volcano plots to improve existing candidates.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
