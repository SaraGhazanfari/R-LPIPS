from __future__ import absolute_import

import numpy as np
import torch
from collections import OrderedDict
import os

from lpips.modular_lpips import LPIPS_Metric, AlexNetFeatureModel, ContrastiveLoss

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class ConstrastiveTrainer:
    def name(self):
        return self.model_name

    def __init__(self, model_path=None, use_gpu=True, is_train=True, lr=.0001, beta1=0.5, train_mode='natural',
                 perturbed_input=None, attack_type=None):

        self.use_gpu = use_gpu
        self.gpu_ids = [DEVICE]
        self.train_mode = train_mode
        self.perturbed_input = perturbed_input
        self.attack_type = attack_type
        self.is_train = is_train
        self.feature_model = AlexNetFeatureModel(path=model_path)
        self.lpips_metric = LPIPS_Metric()

        if self.is_train:  # training mode
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.feature_model.parameters(), lr=lr, betas=(beta1, 0.999))
        else:  # test mode
            self.feature_model.eval()

        if use_gpu:
            self.feature_model.to(self.gpu_ids[0])

    def forward(self, in0):
        return self.feature_model.forward(in0.to(DEVICE))

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        delta = self.generate_attack_on_selected_variable()
        self.forward_train(delta)
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.feature_model.modules():
            if hasattr(module, 'weight') and hasattr(module, 'kernel_size') and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):

        self.input_ref = torch.tensor(data['ref'], device=DEVICE, requires_grad=True)
        self.input_p0 = torch.tensor(data['p0'], device=DEVICE, requires_grad=True)
        self.input_p1 = torch.tensor(data['p1'], device=DEVICE, requires_grad=True)
        self.input_judge = torch.tensor(data['judge'], device=DEVICE)

    def pgd_linf_attack(self, num_iter=50, alpha=1e4, epsilon=8 / 255):
        delta = torch.zeros_like(self.input_ref, requires_grad=True).to(DEVICE)
        for t in range(num_iter):
            loss_total = self.forward_train(delta)
            loss_total.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()

    def pgd_l2_attack(self, num_iter=50, alpha=1e4, epsilon=1.0):
        batch_size = self.input_p0.shape[0]
        delta = torch.zeros_like(self.input_p0, requires_grad=True).to(DEVICE)
        for t in range(num_iter):
            loss_total = self.forward_train(delta)
            loss_total.backward()
            delta.data = delta + alpha * delta.grad.detach().sign()
            delta.grad.zero_()
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta.data = delta * factor.view(-1, 1, 1, 1)

        return delta.detach()

    def generate_attack_on_selected_variable(self):
        if self.attack_type == 'l2':
            delta = self.pgd_l2_attack()

        elif self.attack_type == 'linf':
            delta = self.pgd_linf_attack()

        return delta

    def forward_train(self, delta=0):  # run forward pass

        ref_plus_delta_feature = self.forward(self.input_ref + delta)
        p0_feature = self.forward(self.input_p0)
        p1_feature = self.forward(self.input_p1)
        ref_feature = self.forward(self.input_ref)

        self.acc_r = self.compute_accuracy(self.lpips_metric(ref_plus_delta_feature, p0_feature),
                                           self.lpips_metric(ref_plus_delta_feature, p1_feature), self.input_judge)

        self.loss_total = ContrastiveLoss().forward(ref_feature=ref_feature,
                                                    ref_feature_plus_delta=ref_plus_delta_feature
                                                    , p0_feature=p0_feature, p1_feature=p1_feature,
                                                    judge=self.input_judge)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                               ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def save(self, path, label):
        # if(self.use_gpu):
        #     self.save_network(self.net.modules, path, '', label)
        # else:
        self.save_network(self.feature_model, path, '', label)
        # self.save_network(self.rankLoss.net, path, 'rank', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s' % save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'), flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'), [flag, ], fmt='%i')
