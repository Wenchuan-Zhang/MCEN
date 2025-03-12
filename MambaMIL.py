"""
MambaMIL
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, conf, input_dim,n_classes=2,dropout=None,act='relu',rrt=None,layer=2, rate=10, type="SRMamba"):
    
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(conf.n_features, 512)] #512 >512
        if conf.act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif conf.act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if conf.dropout:
            self._fc1 += [nn.Dropout(conf.dropout)]

        if rrt is not None: 
            self._fc1 += [rrt] 
        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = conf.num_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):
        h = x.float()  # [B, n, 1024] #1 1028 512
        B, N, C = h.size()
        h = self._fc1(h)  # [B, n, 256]

        if self.type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h) # [B, n, K] #attn
        AM = F.softmax(A, dim=1).squeeze(0)  # [B, n]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)
        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]

        Y_prob = F.softmax(logits, dim = 1) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1] 
        logit = torch.sigmoid(logits) 

        return logit, AM, Y_prob,Y_hat 