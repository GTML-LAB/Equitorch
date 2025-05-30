# Equitorch

![Equitorch_logo](./img/logo_wide.png)

*Equitorch* is a modularized package that can be used to flexibly constructing equivariant neural networks.

**[Github Pages](https://github.com/GTML-LAB/Equitorch/tree/main)**

**[Documentation](https://equitorch.readthedocs.io/en/latest/index.html)**

> This package is still under development. 
> We are actively adding more operations, documentations and tutorials.

In this package, we implemented many basic operators that may need in equivariant neural networks, currently including:



### Installation

This package is based on [Pytorch](https://pytorch.org/)(>=2.4), [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)(>=2.4), and [Triton](http://triton-lang.org/)(>=3.2). Please make sure you have already installed the version that fit your device. (It is currently recommended to use `pip` to install the Pytorch-Geometric.)

With these packages installed, you can install *Equitorch* by

```bash
pip install git+https://github.com/GTML-LAB/Equitorch.git
```