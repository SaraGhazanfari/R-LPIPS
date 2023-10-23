# R-LPIPS -  2nd ICML Workshop on New Frontiers in AdvML

### [Paper](https://arxiv.org/pdf/2307.15157) | [Bibtex](#bibtex) | [BAPPS Dataset](https://drive.google.com/file/d/1X3ESpwLdLTyWTC44dBRBp6jY1qpPS1w9/view?usp=share_link)
Robust Learned Perceptual Image Patch Similarity.
This project is applying Adversarial Training to
<a href="https://github.com/richzhang/PerceptualSimilarity" target="_blank">LPIPS model</a>
proposed in: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. In CVPR, 2018.

To adversarially train the model and fine-tune the last linear layer:

```
python train.py --use_gpu --net alex --name alexnet_adv_linf_x0 --train_mode adversarial --perturbed_input x_0 \
--attack_type linf --data_path /path/to/data

```

To adversarially train the model and finetune the alexnet weights as well as the last linear layer:

```
python train.py --train_trunk --use_gpu --net alex --name alexnet_adv_linf_x0_tune --train_mode adversarial 
--perturbed_input x_0 --attack_type linf --data_path /path/to/data

```

* For --train_mode, both "natural" and "adversarial" are available. 
* For --attack_type three options can be used "linf", "l2"
and <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Frequency-Driven_Imperceptible_Adversarial_Attack_on_Semantic_Similarity_CVPR_2022_paper.pdf" target="_blank">
"Semantic Similarity Attack"</a>.
* For --perturbed_input three options are available "x_0" or "x_1" or "x_0/x_1".

To test the robustness of the trained model to adversarial data we have provided two options:
1- Calculating the accuracy of the model on the perturbed test data:

```
python3 test_dataset_model.py --use_gpu --net alex --test_mode adversarial --perturbed_input x_0 --attack_type linf \
--model_path checkpoints/path/to/model.pth --data_path /path/to/data

```

* For test_mode, both "natural" and "adversarial" are available.
* For --attack_type two options can be used "linf" and "l2".
* For --perturbed_input three options are available "x_0" or "x_1" or "x_0/x_1".

2- Loading the trained model and checking its robustness on imagenet-100 validation dataset while 
generating different attacks. In this part we can load two different versions of LPIPS 
and compare their robustness to three attacks suggested below.

```
python evaluate_lpips_robustness.py --data_path path/to/imagenet-100 --attack_type 
linf --first_model_path path/to/first/model --second_model_path path/to/second/model
--target_model_idx 1 --hist_path path/for/hist/to-be-saved 
```

* The  --attack_type can be linf, l2 and aug, which is the BYOL augmentation pipeline.
* The --first_model_path and --second_model_path are the trained LPIPS models, i.e. LPIPS, R-LPIPS or R-LPIPS(lin)
* and R-LPIPS (tune) that we want to compare their robustness.
* the --target_model_idx is the model that we want to perform attack on. It can be any of the first or second model.
*the --hist_y_max, --hist_y_bin_size, --hist_x_max and --hist_x_bin_size are optional and you can set to have more elegant figures.

The trained model for different versions of R-LPIPS is included in the checkpoints directory of the project:
* latest_net_linf_x0.pth: Adversarially trained LPIPS with Linf over x_0
* latest_net_linf_x1.pth: Adversarially trained LPIPS with Linf over x_1
* latest_net_linf_ref.pth: Adversarially trained LPIPS with Linf over reference image (x)
* latest_net_linf_x0_x1.pth: Adversarially trained LPIPS with Linf over x_0 and x_1
* latest_net_linf_x0_tune.pth: Adversarially trained LPIPS in tune mode with Linf over x_0
* latest_net_SSA_x0_tune.pth: Adversarially trained LPIPS with Semantic Similarity Attack (SSA) over x_0

<a name="bibtex"></a>
## Citation
```
@article{ghazanfari2023r,
  title={R-LPIPS: An Adversarially Robust Perceptual Similarity Metric},
  author={Ghazanfari, Sara and Garg, Siddharth and Krishnamurthy, Prashanth and Khorrami, Farshad and Araujo, Alexandre},
  journal={arXiv preprint arXiv:2307.15157},
  year={2023}
}
```
