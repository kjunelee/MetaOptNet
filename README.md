# Meta-learning with differentiable convex optimization
This repository contains the code for the paper:
<br>
**Meta-Learning with Differentiable Convex Optimization**
<br>
Kwonjoon Lee, [Subhransu Maji](https://people.cs.umass.edu/~smaji/), Avinash Ravichandran, [Stefano Soatto](http://web.cs.ucla.edu/~soatto/)   
CVPR 2019 (**Oral**)

### Introduction

Many meta-learning approaches for few-shot learning rely on simple base learners such as nearest-neighbor classifiers. However, even in the few-shot regime, discriminatively trained linear predictors can offer better generalization. We propose to use these predictors as base learners to learn representations for few-shot learning and show they offer better tradeoffs between feature size and performance across a range of few-shot recognition benchmarks. Our objective is to learn feature embeddings that generalize well under a linear classification rule for novel categories. To efficiently solve the objective, we exploit two properties of linear classifiers: implicit differentiation of the optimality conditions of the convex problem and the dual formulation of the optimization problem. This allows us to use high-dimensional embeddings with improved generalization at a modest increase in computational overhead. Our approach, named MetaOptNet, achieves state-of-the-art performance on miniImageNet, tieredImageNet, CIFAR-FS and FC100 few-shot learning benchmarks.

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{lee2019meta,
  title={Meta-Learning with Differentiable Convex Optimization},
  author={Kwonjoon Lee and Subhransu Maji and Avinash Ravichandran and Stefano Soatto},
  booktitle={CVPR},
  year={2019}
}
```

### Acknowledgments

This code is based on the implementations of [**Prototypical Networks**](https://github.com/cyvius96/prototypical-network-pytorch) and [**Dynamic Few-Shot Visual Learning without Forgetting**](https://github.com/gidariss/FewShotWithoutForgetting).

miniImageNet 5-way accuracies
                                                                             
| Method                                     | 1-shot            | 5-shot          |
|--------------------------------------------| ----------------- | --------------- |
| ProtoNet (reported in paper)               | 49.4 ± 0.8 %      | 68.2 ± 0.7 %    |
| ProtoNet (our implementation)              | 52.0 ± 0.6 %      | 68.5 ± 0.5 %    |
| R2D2 (reported in paper)                   | 51.2 ± 0.6 %      | 68.2 ± 0.6 %    |
| R2D2 (our implementation)                  | 54.0 ± 0.6 %      | 70.0 ± 0.5 %    |
| MetaOptNet (ours)                          | 57.7 ± 0.6 %      | 73.6 ± 0.5 %    |

For ProtoNet and R2D2, we used the network architectures used in the original publication. For MetaOptNet, we used ResNet12.
As a note, we use a different training episode composition from the ProtoNet and R2D2 papers: we always trained with 5-way and each class contains 6 query points per episode.
In order to replicate the results, execute the following commands:


Prototypical Networks 1-shot
```
python train.py --gpu 0 --save-path "./experiments/exp_proto_1_shot" --train-way 5 --shot 1 --query 6 --head ProtoNet --network ProtoNet --episodes-per-batch 8
python test.py --gpu 0 --load ./experiments/exp_proto_1_shot/best_model.pth --episode 1000 --way 5 --shot 1 --query 15 --head ProtoNet --network ProtoNet
```
Prototypical Networks 5-shot
```
python train.py --gpu 0 --save-path "./experiments/exp_proto_5_shot" --train-way 5 --shot 5 --query 6 --head ProtoNet --network ProtoNet --episodes-per-batch 8
python test.py --gpu 0 --load ./experiments/exp_proto_5_shot/best_model.pth --episode 1000 --way 5 --shot 5 --query 15 --head ProtoNet --network ProtoNet
```
R2D2 1-shot
```
python train.py --gpu 0 --save-path "./experiments/exp_R2D2_1_shot" --train-way 5 --shot 1 --query 6 --head R2D2 --network R2D2 --episodes-per-batch 8
python test.py --gpu 0 --load ./experiments/exp_R2D2_1_shot/best_model.pth --episode 1000 --way 5 --shot 1 --query 15 --head R2D2 --network R2D2
```

R2D2 5-shot
```
python train.py --gpu 0 --save-path "./experiments/exp_R2D2_5_shot" --train-way 5 --shot 5 --query 6 --head R2D2 --network R2D2 --episodes-per-batch 8
python test.py --gpu 0 --load ./experiments/exp_R2D2_5_shot/best_model.pth --episode 1000 --way 5 --shot 5 --query 15 --head R2D2 --network R2D2
```
MetaOptNet 1-shot
```
python train.py --gpu 0 --save-path "./experiments/exp_MetaOptNet_1_shot" --train-way 20 --shot 1 --query 6 --head SVM --network ResNet --episodes-per-batch 2
python test.py --gpu 0 --load ./experiments/exp_MetaOptNet_1_shot/best_model.pth --episode 1000 --way 5 --shot 1 --query 15 --head SVM --network ResNet
```
MetaOptNet 5-shot
```
python train.py --gpu 0 --save-path "./experiments/exp_MetaOptNet_5_shot" --train-way 15 --shot 5 --query 6 --head SVM --network ResNet --episodes-per-batch 1
python test.py --gpu 0 --load ./experiments/exp_MetaOptNet_5_shot/best_model.pth --episode 1000 --way 5 --shot 5 --query 15 --head SVM --network ResNet
```
