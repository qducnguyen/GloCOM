import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.KNNTM.TopicDistQuant import TopicDistQuant


class KNNTM(nn.Module):
    def __init__(self, vocab_size, num_training_size, training_data, 
                 M_cos_dist_path, M_coo_dist_path,
                 num_topics=50, en_units=200, 
                 alpha=1.0, num_k=20, eta=0.5, rho=0.5):
        super().__init__()


        self.alpha = alpha
        self.num_k = num_k
        self.eta = eta
        self.rho = rho

        M_cos_dist = np.load(M_cos_dist_path)['arr_0']
        M_coo_dist = np.load(M_coo_dist_path)['arr_0']

        self.M_cos_dist = nn.Parameter(torch.from_numpy(M_cos_dist), requires_grad=False)
        self.M_coo_dist = nn.Parameter(torch.from_numpy(M_coo_dist), requires_grad=False)

        self.training_data = training_data
        self.theta_bank = nn.Parameter(torch.zeros(num_training_size, num_topics), requires_grad=False)

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.topic_dist_quant = TopicDistQuant(num_topics, num_topics)

    def get_beta(self):
        return self.fcd1.weight.T

    def encode(self, inputs):
        e1 = F.softplus(self.fc11(inputs))
        e1 = F.softplus(self.fc12(e1))
        return self.mean_bn(self.fc21(e1))

    def decode(self, theta):
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def get_theta(self, inputs):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)
        return softmax_theta
    
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return torch.square(cost)

    def get_aug_x(self, idx, theta):
        # Batch handling
        theta_dist = self.pairwise_euclidean_distance(theta.detach(), self.theta_bank)
        bow_dist = self.rho * self.M_cos_dist[idx] + (1 - self.rho) * self.M_coo_dist[idx]
        fuse_dist = self.eta * theta_dist + (1 - self.eta) * bow_dist

        fuse_dist[torch.arange(idx.size(0)), idx] = float('inf')
        _, topk_indices = torch.topk(fuse_dist, k=self.num_k, dim=1, largest=False)
        neighbors = self.training_data[topk_indices]

        return 1 / self.num_k * self.alpha * torch.sum(neighbors, dim=1)

    def forward(self, idx, inputs, is_aug):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)

        quant_rst = self.topic_dist_quant(softmax_theta)

        recon = self.decode(quant_rst['quantized'])
        loss = self.loss_function(recon, inputs) + quant_rst['loss']
        
        if not is_aug:
            loss = self.loss_function(recon, inputs) + quant_rst['loss']
        else:
            loss = self.loss_function(recon, inputs + self.get_aug_x(idx, softmax_theta)) + quant_rst['loss']

        # Update theta banks
        self.theta_bank[idx] = softmax_theta.clone().detach()

        return {'loss': loss}

    def loss_function(self, recon_x, x):
        loss = -(x * (recon_x).log()).sum(axis=1)
        loss = loss.mean()
        return loss
