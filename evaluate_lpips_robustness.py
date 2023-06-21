import argparse

import jax
import torch
import torchvision

from advertorch.attacks import LinfPGDAttack
from lpips.modular_lpips import AlexNetFeatureModel, LPIPS_Metric
from util.imagenet_dataset import get_dataset
import time
from util.plot_utils import show_adversary_images, plot_histogram
from util.byol_augmentation import postprocess
import torch.nn as nn
from torch import Tensor
import numpy as np

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('mps:0')


class MyMSELoss(nn.MSELoss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MyMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return nn.MSELoss()(input[0], target[0]) + nn.MSELoss()(input[1], target[1]) + nn.MSELoss()(input[2], target[
            2]) + nn.MSELoss()(input[3], target[3]) + nn.MSELoss()(input[4], target[4])


def calculate_distance(x: torch.tensor, x_prime: torch.tensor, lpips_metric: LPIPS_Metric, alexnet: AlexNetFeatureModel,
                       r_alexnet: AlexNetFeatureModel):
    lpips_distance = lpips_metric(alexnet(x), alexnet(x_prime))
    r_lpips_distance = lpips_metric(r_alexnet(x), r_alexnet(x_prime))
    l2_distance = torch.norm(input=x - x_prime, p=2, dim=(0, 1, 2))
    linf_distance = torch.norm(input=x - x_prime, p=float('inf'), dim=(0, 1, 2))
    return lpips_distance, r_lpips_distance, l2_distance, linf_distance


def generate_lp_attack_against_LPIPS_model(lpips_metric, first_feature_model: AlexNetFeatureModel,
                                           second_feature_model: AlexNetFeatureModel,
                                           target_model: AlexNetFeatureModel, show_image=False, threshold=0):
    adversary = LinfPGDAttack(
        target_model, loss_fn=MyMSELoss(), eps=0.05,
        nb_iter=50, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    lpips_list, r_lpips_list, linf_list, l2_list = [], [], [], []

    start_time = time.time()

    for idx, (x, _) in enumerate(dataloader):
        print('{current_idx}/{total_idx}, time elapsed: {delta_time}'.format(current_idx=idx,
                                                                             total_idx=len(dataloader),
                                                                             delta_time=int(
                                                                                 (time.time() - start_time) / 60)))
        x = x.to(DEVICE)
        adv_image = adversary.perturb(x, target_model(x))
        lpips_distance, r_lpips_distance, l2_distance, \
            linf_distance = calculate_distance(x, adv_image, lpips_metric=lpips_metric,
                                               alexnet=first_feature_model, r_alexnet=second_feature_model)
        lpips_list.extend(lpips_distance.tolist())
        r_lpips_list.extend(r_lpips_distance.tolist())
        linf_list.extend(linf_distance.tolist())
        l2_list.extend(l2_distance.tolist())
        if show_image:
            show_adversary_images(x, adv_image, lpips_list, r_lpips_list,
                                  l2_list, linf_list, threshold=threshold)

    return lpips_list, r_lpips_list, linf_list, l2_list


def generate_semantic_attack_against_LPIPS_model_using_byol(lpips_metric: LPIPS_Metric, alexnet: AlexNetFeatureModel,
                                                            r_alexnet: AlexNetFeatureModel):
    rng = jax.random.PRNGKey(0)
    lpips_distance_list, r_lpips_distance_list, l2_distance_list, linf_distance_list = list(), list(), list(), list()
    start_time = time.time()
    for idx, (data, label) in enumerate(dataloader):
        print('{current_idx}/{total_idx}, time elapsed: {delta_time}'.format(current_idx=idx,
                                                                             total_idx=len(dataloader),
                                                                             delta_time=int(
                                                                                 (time.time() - start_time) / 60)))

        inputs = postprocess({'view1': data.numpy().transpose(0, 2, 3, 1),
                              'view2': data.numpy().transpose(0, 2, 3, 1),
                              'labels': label}, rng)
        lpips_distance, r_lpips_distance, l2_distance, linf_distance = \
            calculate_distance(data.to(DEVICE),
                               torch.from_numpy(np.asarray(inputs['view1'].transpose(0, 3, 1, 2))).to(DEVICE),
                               lpips_metric=lpips_metric, alexnet=alexnet, r_alexnet=r_alexnet)

        lpips_distance_list.extend(lpips_distance.tolist())
        r_lpips_distance_list.extend(r_lpips_distance.tolist())
        l2_distance_list.extend(l2_distance.tolist())
        linf_distance_list.extend(linf_distance.tolist())

        lpips_distance, r_lpips_distance, l2_distance, linf_distance = \
            calculate_distance(data.to(DEVICE),
                               torch.from_numpy(np.asarray(inputs['view2'].transpose(0, 3, 1, 2))).to(DEVICE),
                               lpips_metric=lpips_metric, alexnet=alexnet, r_alexnet=r_alexnet)

        lpips_distance_list.extend(lpips_distance.tolist())
        r_lpips_distance_list.extend(r_lpips_distance.tolist())
        l2_distance_list.extend(l2_distance.tolist())
        linf_distance_list.extend(linf_distance.tolist())
    return lpips_distance_list, r_lpips_distance_list, l2_distance_list, linf_distance_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch size', default=50)
    parser.add_argument('--data_path', type=str, help='data path', default=None)
    parser.add_argument('--attack_type', type=str, help='linf, l2, aug', default=None)
    parser.add_argument('--first_model_path', type=str, help='first model path', default=None)
    parser.add_argument('--second_model_path', type=str, help='second model path', default=None)
    parser.add_argument('--target_model_idx', type=int, help='idx of the target model to generate attack against',
                        default=None)
    parser.add_argument('--hist_path', type=str, help='destination path for the histogram',
                        default='hist.pdf')

    args = parser.parse_args()

    model = torchvision.models.resnet50(weights="DEFAULT")
    val_dataset, dataloader = get_dataset(batch_size=args.batch_size, num_workers=1, data_path=args.data_path,
                                          split='val')

    lpips_metric = LPIPS_Metric().eval()
    first_feature_model = AlexNetFeatureModel(path=args.first_model_path).eval()
    second_feature_model = AlexNetFeatureModel(path=args.second_model_path).eval()

    if args.attack_type == 'aug':
        lpips_list, r_lpips_list, linf_list, l2_list = generate_semantic_attack_against_LPIPS_model_using_byol(
            lpips_metric, first_feature_model, second_feature_model)
    elif args.target_model_idx == 1:
        lpips_list, r_lpips_list, linf_list, l2_list = generate_lp_attack_against_LPIPS_model(
            lpips_metric=lpips_metric,
            second_feature_model=second_feature_model,
            first_feature_model=first_feature_model,
            target_model=first_feature_model,
            show_image=False,
            threshold=0)

    else:  # args.target_model_idx == 2
        lpips_list, r_lpips_list, linf_list, l2_list = generate_lp_attack_against_LPIPS_model(
            lpips_metric=lpips_metric,
            second_feature_model=second_feature_model,
            first_feature_model=first_feature_model,
            target_model=second_feature_model,
            show_image=False,
            threshold=0)

    plot_histogram(lpips_list, r_lpips_list, save_path=args.hist_path)
