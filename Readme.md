# Debiasing Model Updates for Improving Personalized Federated Training

This is implementation of [Debiasing Model Updates for Improving Personalized Federated Training](https://icml.cc/Conferences/2021/AcceptedPapersInitial).

### Requirements

Please install the required packages. The code is compiled with Python 3.7 dependencies in a virtual environment via

```pip install -r requirements.txt```

### Instructions

An example code for CIFAr-10, ACID, 5 class per device setting is given. Run

```example_code_cifar10.py```

The code,

- Constructs a federated dataset,

- Trains all methods,

- Plots the average test accuracy vs. rounds convergence curves.