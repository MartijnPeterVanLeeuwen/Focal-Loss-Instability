# A Note on the Stability of the Focal Loss

This repository provides additional information regarding our paper on the stability of the Focal loss. The Focal loss is a popular loss function designed to deal with class-imbalanced datasets, and is parameterized by a focusing parameter $\gamma$ that determines the degree to which easy examples are downscaled. In our [paper](https://openreview.net/forum?id=eCYActnGbu), we addressed a focal loss instability that can occur whenever this focusing parameter $\gamma$ is set to a value between 0 and 1 due to a singularity in the derivative of the Focal loss.

To prove that this instability is not only a theoretical issue, we have demonstrated that the instability can be detected in a two-binary classification and one segmentation experiment. The code for these experiments is included in this repository, along with the generated results, which can be found [here](https://github.com/MartijnPeterVanLeeuwen/Focal-Loss-Instability/tree/main/Experiment_Results).

In this paper, we also provide a solution to the instability by adding a smoothing constant to the modulating factor that downscales the cross-entropy loss. We have repeated the experiments with this modified version of the focal loss and have shown that no more instabilities are found. The modified version of the Focal loss can be found in this repository as well by following this [link](https://github.com/MartijnPeterVanLeeuwen/Focal-Loss-Instability/blob/main/CODE/Stabilized_Focal_loss.py). 

In our paper, we have used a smoothing constant with a value of 1e-3, where we have found that it stabilizes model training for $\gamma$ values as small as 0.1. However, it could be possible that $\gamma$ values smaller than 0.1 still lead to instabilities with this smoothing constant value. If instability is still encountered, the value of the smoothing constant should be increased.

We have also provided some code to demonstrate that "NaN" values can be detected when calculating the gradient of the Focal loss, when no smoothing constant is used. This code can be found [here](https://github.com/MartijnPeterVanLeeuwen/Focal-Loss-Instability/blob/main/CODE/Gradient_analysis/Numerical_Computation_Gradient.py)

If you make use of this focal loss stabilization method, please cite our paper! 

```sh
@article{
leeuwen2025a,
title={A Note On The Stability Of The Focal Loss},
author={Martijn P. van Leeuwen and Koen V. Haak and Gorkem Saygili and Eric O. Postma and L.L. Sharon Ong},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=eCYActnGbu},
note={}
}
```
