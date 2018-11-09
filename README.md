# Meta-learning with differentiable convex optimization
The following are the miniImageNet few-shot accuracies of ProtoNet (Snell et al., NIPS 2017), Meta-learning with differentiable closed-form solvers — R2D2 (Bertinetto et al., arXiv 2018), and MetaOptNet (ours) ran on our codebase.

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
