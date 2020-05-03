## yoto_class_balanced_loss
Unofficial implementation of YOTO (You Only Train Once) applied to Class balanced loss<br>

YOU ONLY TRAIN ONCE: LOSS-CONDITIONAL TRAINING OF DEEP NETWORKS<br>
https://openreview.net/pdf?id=HyxY6JHKwr<br>

Class-Balanced Loss Based on Effective Number of Samples<br>
https://arxiv.org/abs/1901.05555<br>

## Overview
Image classification was performed by applying YOTO to Class balanced loss.
By using YOTO, I was able to select a model with good performance in the Major class or a model with good performance in the Minor class at the time of testing with only one model.<br>

## Verification
I converted Cifar 10 to unbalanced data and used it for validation. The number of data in classes 1, 3, 5, 7, and 9 of the training data was reduced to 1/10. The number of test data was not changed.

For the base model, I used ResNet18.
By combining YOTO and Class balanced loss, we trained a model in which β, a hyper-parameter for minor class weights, can be changed at test time.<br>

The classification accuracy for each data is shown in the figure below.<br>
<img src="https://github.com/statsu1990/yoto_class_balanced_loss/blob/master/results/case1_summary.png" width="320px">

By changing the beta at the time of testing, we can see that one model can choose a model with good performance in the Major class or a model with good performance in the Minor class.<br>
The performance of YOTO is high when 1-β is small. This is strange and I'm not sure why.
When 1-β is large, the ratio of the weights of the Major and Minor classes is nearly 20 times larger. It is my personal expectation that learning becomes unstable when the hyperparameters are fixed, but it may be stable when YOTO is used.<br>

## More details
https://st1990.hatenablog.com/entry/2020/05/04/012738
