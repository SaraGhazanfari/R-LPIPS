# R-LPIPS
Robust Learned Perceptual Image Patch Similarity. 
This project is applying Adversarial Training to 
<a href="https://github.com/richzhang/PerceptualSimilarity" target="_blank">LPIPS model</a> 
proposed in:

The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. In CVPR, 2018.

To adversarially train the model and learn the last linear layer:
```
python3 train.py --use_gpu --net alex --name alexnet_adv_linf_x0 --train_mode adversarial --perturbed_input x_0 \
--attack_type linf --data_path /path/to/data

```
To adversarially train the model, learn the last linear layer and finetune alexnet weights:
```
python3 ./train.py --train_trunk --use_gpu --net alex --name alexnet_adv_linf_x0_tune --train_mode adversarial 
--perturbed_input x_0 --attack_type linf --data_path /path/to/data

```
For --train_mode, both "natural" and "adversarial" are available.
For --attack_type three options can be used "linf", "l2" and <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Frequency-Driven_Imperceptible_Adversarial_Attack_on_Semantic_Similarity_CVPR_2022_paper.pdf" target="_blank">Semantic Similarity Attack</a>. 
For --perturbed_input three options are available "x_0" or "x_1" or "x_0/x_1".

To test the robustness of trained model to adversarial data:
```
python3 test_dataset_model.py --use_gpu --net alex --test_mode adversarial --perturbed_input x_0 --attack_type linf \
--model_path ./checkpoints/path/to/model.pth --data_path /path/to/data

```
For test_mode, both "natural" and "adversarial" are available.
For --attack_type two options can be used "linf" and "l2". 
For --perturbed_input three options are available "x_0" or "x_1" or "x_0/x_1".
