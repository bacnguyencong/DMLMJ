# DMLMJ
Supervised distance metric learning through maximization of the Jeffrey divergence

** How to learn a linear transformation $A$ ?**

$$\underset{\vec{A}\in\mathbb{R}^{D\times m}}{\arg\max} \quad f(\vec{A}) = \text{KL}(P_{\vec{A}}, Q_{\vec{A}}) + \text{KL}(Q_{\vec{A}}, P_{\vec{A}})$$

<img src="data/1-s2.0-S0031320316303600-gr2_lrg.jpg" style="max-width:100%; width: 50%"> <img src="data/1-s2.0-S0031320316303600-gr3_lrg.jpg" style="max-width:100%; width: 50%">


### Prerequisites
This has been tested using MATLAB 2010A and later on Windows and Linux (Mac should be fine).

## Installation
Download the repository into the directory of your choice. Then within MATLAB go to file >> Set path... and add the directory containing "DML-dc" to the list (if it isn't already). That's it.

## Usage
Please run (inside the matlab console)
```matlab
demo  % demo of DML-dc
```
## Authors
* [Bac Nguyen Cong](https://github.com/bacnguyencong)

## Acknowledgments
If you find this code useful in your research, please consider citing:
``` bibtex
@Article{Nguyen2018,
  Title       = {An approach to supervised distance metric learning based on difference of convex functions programming},
  Author      = {Bac Nguyen and De Baets, Bernard},
  Journal     = {Pattern Recognition},
  Year        = {2018}
}
```
