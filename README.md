torch_itl
=========

Algorithms for solving integral loss minimization problems. Currently, we handle the following problems:

- Joint Quantile Regression
- Emotion Transfer with squared loss
- ITL with general loss provided by the user based on pytorch autodiff

Installation (development version)
==================================
To install the package, clone it and run
`$ pip install -e .`

This installs automatically the non-satisfied dependencies in `./requirements.txt`

Demos
=====

You can find examples for both problems in the demo section, for both [infinite quantile regression](https://github.com/allambert/torch_itl/tree/release/demos/quantile) and [emotion transfer](https://github.com/allambert/torch_itl/tree/release/demos/emotion_transfer).


Cite
====

If you use this code, please cite the corresponding work:

```
@inproceedings{brault2019infinite,
  title={Infinite task learning in rkhss},
  author={Brault, Romain and Lambert, Alex and Szab{\'o}, Zolt{\'a}n and Sangnier, Maxime and d’Alch{\'e}-Buc, Florence},
  booktitle={The 22nd International Conference on Artificial Intelligence and Statistics},
  pages={1294--1302},
  year={2019},
  organization={PMLR}
}

@preprint{emo_transfer_lambert,
  title={Emotion Transfer Using Vector-Valued Infinite Task Learning},
  author={Lambert, Alex and Parekh, Sanjeel and Szab{\'o}, Zolt{\'a}n and d’Alch{\'e}-Buc, Florence},
  year={2021}
}
