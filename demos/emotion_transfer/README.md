Emotion Transfer
================

We cast an emotion transfer problem as an infinite task learning one with vectorial outputs, as proposed in [Emotion Transfer Using Vector-Valued Infinite Task Learning](https://allambert.github.io/files/pdf/emo_transfer.pdf). We apply it on the [KDEF](https://www.kdef.se/) and [RaFD](http://www.socsci.ru.nl:8180/RaFD2/RaFD) datasets.

The folder contains the following script:

- `cross_validation.py`: cross validation loop used for selecting the best values of hyperparameters.

- `dim_reduction.py`: Experiments with non-identity output kernel matrix aimed at dimensionality reduction.

- `missing_data`: Experiments evauating the performances of the estimator in the presence of missing data.

- `visualization.py`: Script allowing to generate figures presented in the paper.
