Quantile regression
================

We jointly solve a joint quantile regression problem as an infinite task learning one, as proposed in [Infinite Task Learning in RKHSs](https://allambert.github.io/files/pdf/paper_ITL.pdf). We also allow the kernels involved to be learnable, by defining them as composition between a neural network and a scalar kernel.

The folder contains the following script:

- `demo_synthetic.py`: Proof of work on synthetic data, using a deep kernel on the inputs
