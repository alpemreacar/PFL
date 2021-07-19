# Debiasing Model Updates for Improving Personalized Federated Training

This is implementation of [Debiasing Model Updates for Improving Personalized Federated Training](http://proceedings.mlr.press/v139/acar21a.html).

### Requirements

Please install the required packages. The code is compiled with Python 3.7 dependencies in a virtual environment via

```pip install -r requirements.txt```

### Instructions

An example code for CIFAr-10, ACID, 5 class per device setting is given. Run

```python cifar10_ACID.py```

The code,

- Constructs a federated dataset,

- Trains all methods,

- Plots the average test accuracy vs. rounds convergence curves.

### Citation

```
@InProceedings{pmlr-v139-acar21a,
  title = {Debiasing Model Updates for Improving Personalized Federated Training},
  author = {Acar, Durmus Alp Emre and Zhao, Yue and Zhu, Ruizhao and Matas, Ramon and Mattina, Matthew and Whatmough, Paul and Saligrama, Venkatesh},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {21--31},
  year = {2021},
  editor = {Meila, Marina and Zhang, Tong},
  volume = {139},
  series = {Proceedings of Machine Learning Research},
  month = {18--24 Jul},
  publisher = {PMLR},
  pdf =  {http://proceedings.mlr.press/v139/acar21a/acar21a.pdf},
  url =  {http://proceedings.mlr.press/v139/acar21a.html}
}
```
