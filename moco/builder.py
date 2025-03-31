# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).cuda() # + N * torch.distributed.get_rank()).cuda() № 
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        
class MoCo_ResNet_Custom_Channels(MoCo):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, input_channels=3):
        # Вызываем конструктор родительского класса
        super().__init__(base_encoder, dim, mlp_dim, T)
        
        # Сохраняем оригинальные параметры первого сверточного слоя
        out_channels = self.base_encoder.conv1.out_channels
        kernel_size = self.base_encoder.conv1.kernel_size
        stride = self.base_encoder.conv1.stride
        padding = self.base_encoder.conv1.padding
        
        # Создаем новые сверточные слои с нужным количеством входных каналов
        self.base_encoder.conv1 = nn.Conv2d(
            input_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        self.momentum_encoder.conv1 = nn.Conv2d(
            input_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Переинициализируем веса momentum encoder
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
            
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
            
    # def _build_projector_and_predictor_mlps(self, dim, mlp_dim, geo_channels):
    #     hidden_dim = self.base_encoder.fc.weight.shape[1]
    #     del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer
        
    #     # projectors
    #     self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
    #     self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

    #     # predictor
    #     self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        
    #     # self.geo_channels = geo_channels
    #     # Изменяем первый слой для поддержки заданного количества входных каналов
    #     in_channels = self.base_encoder.conv1.in_channels
    #     self.base_encoder.conv1 = nn.Conv2d(geo_channels, in_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    #     # Также изменяем первый слой в моментном энкодере
    #     self.momentum_encoder.conv1 = nn.Conv2d(geo_channels, in_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output











# # Copyright (c) Facebook, Inc. and its affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import torch
# import torch.nn as nn
# from torchvision import models


# class MoCo(nn.Module):
#     """
#     Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
#     https://arxiv.org/abs/1911.05722
#     """
#     def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
#         """
#         dim: feature dimension (default: 256)
#         mlp_dim: hidden dimension in MLPs (default: 4096)
#         T: softmax temperature (default: 1.0)
#         """
#         super(MoCo, self).__init__()

#         self.T = T

#         # build encoders
#         self.base_encoder = base_encoder(num_classes=mlp_dim)
#         self.momentum_encoder = base_encoder(num_classes=mlp_dim)

#         self._build_projector_and_predictor_mlps(dim, mlp_dim)

#         for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
#             param_m.data.copy_(param_b.data)  # initialize
#             param_m.requires_grad = False  # not update by gradient

#     def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
#         mlp = []
#         for l in range(num_layers):
#             dim1 = input_dim if l == 0 else mlp_dim
#             dim2 = output_dim if l == num_layers - 1 else mlp_dim

#             mlp.append(nn.Linear(dim1, dim2, bias=False))

#             if l < num_layers - 1:
#                 mlp.append(nn.BatchNorm1d(dim2))
#                 mlp.append(nn.ReLU(inplace=True))
#             elif last_bn:
#                 # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
#                 # for simplicity, we further removed gamma in BN
#                 mlp.append(nn.BatchNorm1d(dim2, affine=False))

#         return nn.Sequential(*mlp)

#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         pass

#     @torch.no_grad()
#     def _update_momentum_encoder(self, m):
#         """Momentum update of the momentum encoder"""
#         for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
#             param_m.data = param_m.data * m + param_b.data * (1. - m)

#     def contrastive_loss(self, q, k):
#         # normalize
#         q = nn.functional.normalize(q, dim=1)
#         k = nn.functional.normalize(k, dim=1)
#         # gather all targets
#         # k = concat_all_gather(k)
#         # Einstein sum is more intuitive
#         logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
#         N = logits.shape[0]  # batch size per GPU
#         labels = (torch.arange(N, dtype=torch.long)).cuda() # + N * torch.distributed.get_rank()).cuda()
#         return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

#     def forward(self, x1, x2, m):
#         """
#         Input:
#             x1: first views of images
#             x2: second views of images
#             m: moco momentum
#         Output:
#             loss
#         """

#         # compute features
#         q1 = self.predictor(self.base_encoder(x1))
#         q2 = self.predictor(self.base_encoder(x2))

#         with torch.no_grad():  # no gradient
#             self._update_momentum_encoder(m)  # update the momentum encoder

#             # compute momentum features as targets
#             k1 = self.momentum_encoder(x1)
#             k2 = self.momentum_encoder(x2)

#         return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


# class MoCo_ResNet(MoCo):
#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         hidden_dim = self.base_encoder.fc.weight.shape[1]
#         del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

#         # projectors
#         self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
#         self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

#         # predictor
#         self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


# class MoCo_ViT(MoCo):
#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         hidden_dim = self.base_encoder.head.weight.shape[1]
#         del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

#         # projectors
#         self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
#         self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

#         # predictor
#         self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output


# class CustomResNet(MoCo):
#     def __init__(self, geo_channels, base_encoder=models.resnet50, dim=256, mlp_dim=4096, T=1.0, **kwargs):
#         # Передаем базовый энкодер и другие параметры в родительский класс MoCo
#         super(CustomResNet, self).__init__(base_encoder=base_encoder, dim=dim, mlp_dim=mlp_dim, T=T, **kwargs)

#         self.geo_channels = geo_channels
        
#         # Изменяем первый слой для поддержки заданного количества входных каналов
#         in_channels = self.base_encoder.conv1.in_channels
#         self.base_encoder.conv1 = nn.Conv2d(geo_channels, in_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#         # Также изменяем первый слой в моментном энкодере
#         self.momentum_encoder.conv1 = nn.Conv2d(geo_channels, in_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# #         # Строим проекторы и предсказатели
#         self._build_projector_and_predictor_mlps(dim, mlp_dim)

#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         # Получаем оригинальный fc слой
#         original_fc = self.base_encoder.fc
        
#         # Получаем размер входа оригинального fc слоя
#         hidden_dim = original_fc.in_features

#         # Удаляем оригинальный fc слой
#         del self.base_encoder.fc, self.momentum_encoder.fc  

#         # Создаем проекторы
#         self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
#         self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

#         # Создаем предсказатель
#         self.predictor = self._build_mlp(2, dim, mlp_dim, dim, use_batchnorm=False)

#     def _build_mlp(self, num_layers, in_dim, hidden_dim, out_dim, use_batchnorm=True):
#         layers = []
#         for _ in range(num_layers):
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             if use_batchnorm:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.ReLU(inplace=True))
#             in_dim = hidden_dim
#         layers.append(nn.Linear(hidden_dim, out_dim))
#         return nn.Sequential(*layers)

#     def forward(self, x1, x2, m):
#         """
#         Input:
#             x1: first views of images
#             x2: second views of images
#             m: moco momentum
#         Output:
#             loss
#         """
#         # compute features
#         q1 = self.predictor(self.base_encoder(x1))
#         q2 = self.predictor(self.base_encoder(x2))

#         with torch.no_grad():  # no gradient
#             self._update_momentum_encoder(m)  # update the momentum encoder

#             # compute momentum features as targets
#             k1 = self.momentum_encoder(x1)
#             k2 = self.momentum_encoder(x2)

#         return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)





# # Пример вызова экземпляра нейросети
# geo_channels = 5  # Замените на нужное количество гео каналов
# model = CustomResNet(geo_channels=geo_channels, base_encoder=models.resnet50, dim=256, mlp_dim=4096, T=0.07)





# import torch
# import torch.nn as nn
# import torchvision.models as torchvision_models

# class CustomResNet(nn.Module):
#     def __init__(self, base_model, in_channels=7, num_classes=256):
#         super(CustomResNet, self).__init__()
#         self.base_model = base_model(num_classes=num_classes)
        
#         old_conv = self.base_model.conv1
#         self.base_model.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             padding=old_conv.padding,
#             bias=False
#         )
#         self.fc = self.base_model.fc

#     def forward(self, x):
#         return self.base_model(x)

# class MoCo(nn.Module):
#     def __init__(self, base_encoder, in_channels=7, dim=256, mlp_dim=4096, T=1.0):
#         super(MoCo, self).__init__()
#         self.T = T

#         self.base_encoder = CustomResNet(base_encoder, in_channels=in_channels, num_classes=mlp_dim)
#         self.momentum_encoder = CustomResNet(base_encoder, in_channels=in_channels, num_classes=mlp_dim)

#         self._build_projector_and_predictor_mlps(dim, mlp_dim)

#         for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
#             param_m.data.copy_(param_b.data)
#             param_m.requires_grad = False

#     def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
#         mlp = []
#         for l in range(num_layers):
#             dim1 = input_dim if l == 0 else mlp_dim
#             dim2 = output_dim if l == num_layers - 1 else mlp_dim

#             mlp.append(nn.Linear(dim1, dim2, bias=False))

#             if l < num_layers - 1:
#                 mlp.append(nn.BatchNorm1d(dim2))
#                 mlp.append(nn.ReLU(inplace=True))
#             elif last_bn:
#                 mlp.append(nn.BatchNorm1d(dim2, affine=False))

#         return nn.Sequential(*mlp)

#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         pass

#     @torch.no_grad()
#     def _update_momentum_encoder(self, m):
#         for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
#             param_m.data = param_m.data * m + param_b.data * (1. - m)

#     def contrastive_loss(self, q, k):
#         q = nn.functional.normalize(q, dim=1)
#         k = nn.functional.normalize(k, dim=1)
        
#         logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
#         N = logits.shape[0]
#         labels = torch.arange(N, dtype=torch.long).cuda()
        
#         return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

#     def forward(self, x1, x2, m):
#         q1 = self.predictor(self.base_encoder(x1))
#         q2 = self.predictor(self.base_encoder(x2))

#         with torch.no_grad():
#             self._update_momentum_encoder(m)
#             k1 = self.momentum_encoder(x1)
#             k2 = self.momentum_encoder(x2)

#         return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

# class MoCo_ResNet(MoCo):
    
#     def __init__(self, base_encoder, in_channels=7, dim=256, mlp_dim=4096, T=1.0):
#         super(MoCo_ResNet, self).__init__(base_encoder, in_channels=in_channels, dim=dim, mlp_dim=mlp_dim, T=T)

#         # Изменение первого сверточного слоя для base_encoder
#         old_conv = self.base_encoder.base_model.conv1
#         self.base_encoder.base_model.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             padding=old_conv.padding,
#             bias=False
#         )

#         # Изменение первого сверточного слоя для momentum_encoder
#         old_conv_momentum = self.momentum_encoder.base_model.conv1
#         self.momentum_encoder.base_model.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=old_conv_momentum.out_channels,
#             kernel_size=old_conv_momentum.kernel_size,
#             stride=old_conv_momentum.stride,
#             padding=old_conv_momentum.padding,
#             bias=False
#         )
    
    
#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         hidden_dim = self.base_encoder.fc.weight.shape[1]
#         del self.base_encoder.fc, self.momentum_encoder.fc

#         self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
#         self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
#         self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)




        
# class MoCo_ViT(MoCo):
#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         hidden_dim = self.base_encoder.head.weight.shape[1]
#         del self.base_encoder.head, self.momentum_encoder.head

#         self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
#         self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
#         self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

# @torch.no_grad()
# def concat_all_gather(tensor):
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#     output = torch.cat(tensors_gather, dim=0)
#     return output

# def create_moco_model(arch='resnet50', in_channels=7, dim=256, mlp_dim=4096, T=1.0):
#     base_encoder = torchvision_models.__dict__[arch]
    
#     model = MoCo_ResNet(
#         base_encoder,
#         in_channels=in_channels,
#         dim=dim,
#         mlp_dim=mlp_dim,
#         T=T
#     )
    
#     return model



# # # Copyright (c) Facebook, Inc. and its affiliates.
# # # All rights reserved.

# # # This source code is licensed under the license found in the
# # # LICENSE file in the root directory of this source tree.

# # import torch
# # import torch.nn as nn


# # class MoCo(nn.Module):
# #     """
# #     Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
# #     https://arxiv.org/abs/1911.05722
# #     """
# #     def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
# #         """
# #         dim: feature dimension (default: 256)
# #         mlp_dim: hidden dimension in MLPs (default: 4096)
# #         T: softmax temperature (default: 1.0)
# #         """
# #         super(MoCo, self).__init__()

# #         self.T = T

# #         # build encoders
# #         self.base_encoder = base_encoder(num_classes=mlp_dim)
# #         self.momentum_encoder = base_encoder(num_classes=mlp_dim)

# #         self._build_projector_and_predictor_mlps(dim, mlp_dim)

# #         for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
# #             param_m.data.copy_(param_b.data)  # initialize
# #             param_m.requires_grad = False  # not update by gradient

# #     def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
# #         mlp = []
# #         for l in range(num_layers):
# #             dim1 = input_dim if l == 0 else mlp_dim
# #             dim2 = output_dim if l == num_layers - 1 else mlp_dim

# #             mlp.append(nn.Linear(dim1, dim2, bias=False))

# #             if l < num_layers - 1:
# #                 mlp.append(nn.BatchNorm1d(dim2))
# #                 mlp.append(nn.ReLU(inplace=True))
# #             elif last_bn:
# #                 # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
# #                 # for simplicity, we further removed gamma in BN
# #                 mlp.append(nn.BatchNorm1d(dim2, affine=False))

# #         return nn.Sequential(*mlp)

# #     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
# #         pass

# #     @torch.no_grad()
# #     def _update_momentum_encoder(self, m):
# #         """Momentum update of the momentum encoder"""
# #         for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
# #             param_m.data = param_m.data * m + param_b.data * (1. - m)

# #     def contrastive_loss(self, q, k):
# #         # normalize
# #         q = nn.functional.normalize(q, dim=1)
# #         k = nn.functional.normalize(k, dim=1)
# #         # gather all targets
# #         # k = concat_all_gather(k)
# #         # Einstein sum is more intuitive
# #         logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
# #         N = logits.shape[0]  # batch size per GPU
# #         labels = (torch.arange(N, dtype=torch.long)).cuda() # + N * torch.distributed.get_rank()).cuda()
# #         return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

# #     def forward(self, x1, x2, m):
# #         """
# #         Input:
# #             x1: first views of images
# #             x2: second views of images
# #             m: moco momentum
# #         Output:
# #             loss
# #         """

# #         # compute features
# #         q1 = self.predictor(self.base_encoder(x1))
# #         q2 = self.predictor(self.base_encoder(x2))

# #         with torch.no_grad():  # no gradient
# #             self._update_momentum_encoder(m)  # update the momentum encoder

# #             # compute momentum features as targets
# #             k1 = self.momentum_encoder(x1)
# #             k2 = self.momentum_encoder(x2)

# #         return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


# # class MoCo_ResNet(MoCo):
# #     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
# #         hidden_dim = self.base_encoder.fc.weight.shape[1]
# #         del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

# #         # projectors
# #         self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
# #         self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

# #         # predictor
# #         self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


# # class MoCo_ViT(MoCo):
# #     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
# #         hidden_dim = self.base_encoder.head.weight.shape[1]
# #         del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

# #         # projectors
# #         self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
# #         self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

# #         # predictor
# #         self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# # # utils
# # @torch.no_grad()
# # def concat_all_gather(tensor):
# #     """
# #     Performs all_gather operation on the provided tensors.
# #     *** Warning ***: torch.distributed.all_gather has no gradient.
# #     """
# #     tensors_gather = [torch.ones_like(tensor)
# #         for _ in range(torch.distributed.get_world_size())]
# #     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

# #     output = torch.cat(tensors_gather, dim=0)
# #     return output
