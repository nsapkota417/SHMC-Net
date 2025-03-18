import torch
import torch.nn as nn
from torch import Tensor
import einops
import random
from typing import Dict, Iterable, Callable

from networks.my_shufflenet_v2_x2_0 import shufflenet_v2_x2_0
from networks.my_densenet_121 import densenet121
from networks.my_resnet_18 import resnet18, resnet34, resnet50


def model_stages_shapes(name):
    if name in ('resnet18', 'resnet34'):
        shapes = [66, 64, 128, 256]
    elif name in ('resnet50'):
        shapes = [64, 256, 512, 1024]
    elif name in ['shufflenet_v2_x2_0']:
        shapes = [24, 244, 488, 976]
        
    return shapes


def change_first_conv(model, in_channels):
    
    named_children = list(model.named_children())
    names = [named_children[0][0]]
    m = named_children[0][1]

    while not isinstance(m, torch.nn.Conv2d):
        named_children = list(m.named_children())
        names.append(named_children[0][0])
        m = named_children[0][1]

    if in_channels != m.in_channels:
        l1 = nn.Conv2d(in_channels, 
                             m.out_channels, 
                             kernel_size=m.kernel_size,
                             stride=m.stride, 
                             padding=m.padding,
                             bias=False if m.bias is None else True)

        m_toset = model
        for m_name in names[:-1]:
            m_toset = m_toset.__getattr__(m_name)

        m_toset.__setattr__(names[-1], l1)
        print(f'\tModified input channel on 1st conv layer, {m.in_channels} >> {in_channels}')

    return model

def change_linear_layer(model, out_channels):
    named_children = list(model.named_children())
    names = [named_children[-1][0]]
    m = named_children[-1][-1]


    while not isinstance(m, nn.Linear):
        named_children = list(m.named_children())
        names.append(named_children[-1][0])
        m = named_children[-1][1]

    last_lin = nn.Linear(in_features=m.in_features,
                         out_features = out_channels,
                         bias=False if m.bias is None else True)

    m_toset = model
    for m_name in names[:-1]:
        m_toset = m_toset.__getattr__(m_name)

    m_toset.__setattr__(names[-1], last_lin)
    print(f'\tModified output channels on last classifier layer, {m.out_features} >> {out_channels}')

    return model


def get_modified_model(backbone_name, in_channels, out_channels, pretrained_weights):
    
    if pretrained_weights == 'I':
        model = globals()[backbone_name](weights='IMAGENET1K_V1') 
        print(f'Using Imagenet Pretrained Weights.')

    else:
        model = globals()[backbone_name]()
    
    if 'mnist' in pretrained_weights:
        print("Pretrained Weights provide, adjusting layers accordingly.")
        model = change_first_conv(model, in_channels=1)
        model = change_linear_layer(model, 10)
        model.load_state_dict(torch.load(pretrained_weights))
        print(f'\tLoading Pretrained weights from: {pretrained_weights}')
    
    print("Adjusting layers for Dataset:")
    model = change_first_conv(model, in_channels=in_channels)
    model = change_linear_layer(model, out_channels=out_channels)
    return model




class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
    
class classifierModel(nn.Module):
    
    def __init__(self, cnf):
        super().__init__()
        self.cnf = cnf
        self.model1 = get_modified_model(cnf.train.backbone,
                       cnf.train.in_channels-1 if cnf.train.in_channels in (2,4) 
                                         else cnf.train.in_channels,
                       cnf.train.out_channels,
                       cnf.train.pretrained_weights)
        
        if cnf.train.in_channels in (2,4):
            self.model2 = get_modified_model(cnf.train.backbone,
                           1,
                           cnf.train.out_channels,
                           cnf.train.pretrained_weights)
        
        self.relu = nn.ReLU(True)
        self.clasfInputCh = self.get_classifier_input_ch()
        self.stages_shape = model_stages_shapes(cnf.train.backbone)
        self.linear = nn.Linear(self.clasfInputCh*2 if self.cnf.train.feature_merge == 'C' and 
                                cnf.train.in_channels in (2,4) else self.clasfInputCh, 
                                cnf.train.out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cnf.train.dropout)
        

        self.fusion_conv1 = nn.Sequential(nn.Conv2d(self.stages_shape[0], self.stages_shape[0], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[0]), nn.ReLU(True))
        
        self.fusion_conv1_msk = nn.Sequential(nn.Conv2d(self.stages_shape[0], self.stages_shape[0], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[0]), nn.ReLU(True))
        
        self.fusion_conv2 = nn.Sequential(nn.Conv2d(self.stages_shape[1], self.stages_shape[1], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[1]), nn.ReLU(True))
        
        self.fusion_conv2_msk = nn.Sequential(nn.Conv2d(self.stages_shape[1], self.stages_shape[1], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[1]), nn.ReLU(True))
        
        self.fusion_conv3 = nn.Sequential(nn.Conv2d(self.stages_shape[2], self.stages_shape[2], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[2]), nn.ReLU(True))
        
        self.fusion_conv3_msk = nn.Sequential(nn.Conv2d(self.stages_shape[2], self.stages_shape[2], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[2]), nn.ReLU(True))
        
        self.fusion_conv4 = nn.Sequential(nn.Conv2d(self.stages_shape[3], self.stages_shape[3], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[3]), nn.ReLU(True))
        
        self.fusion_conv4_msk = nn.Sequential(nn.Conv2d(self.stages_shape[3], self.stages_shape[3], kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(self.stages_shape[3]), nn.ReLU(True))
    
    
    def get_classifier_input_ch(self):
        backbone = self.cnf.train.backbone
        if backbone == 'densenet121': 
            in_ch = self.model1.classifier.in_features
        else:
            in_ch = self.model1.fc.in_features

        return in_ch
    
    
    def forward_debug(self, x, iter_track):
        if self.cnf.train.in_channels in (2,4):
            # apply feature fusion model 
            if self.cnf.train.in_channels == 2:
                im_ch_count = 1
            else:
                im_ch_count = 3

            image = x[:, 0:im_ch_count]
            mask = x[:, -1:]

            # img_feature_object = FeatureExtractor(self.model1, layers = ['conv5'])
            # msk_feature_object = FeatureExtractor(self.model2, layers = ['conv5'])

            print(f'org input: {image.shape}')
            f1_img = self.model1.forward1(image)
            print(f'after fwd 1: {f1_img.shape}')
            f1_img = self.model1.forward2(f1_img)
            print(f'after fwd 2: {f1_img.shape}')
            f1_img = self.model1.forward3(f1_img)
            print(f'after fwd 3: {f1_img.shape}')
            f1_img = self.model1.forward4(f1_img)
            print(f'after fwd 4: {f1_img.shape}')
            f1_img = self.model1.forward5(f1_img)
            print(f'after fwd 5: {f1_img.shape}')
            final_features = f1_img
        
            # fm_image = img_feature_object(image)['conv5']
            # fm_mask = msk_feature_object(mask)['conv5']
            # if self.cnf.train.feature_merge == 'S':
            #     final_features = fm_image+fm_mask
            # else:
            #     final_features = torch.cat((fm_image,fm_mask), dim=1)


        else:
            if self.cnf.train.backbone == 'densenet121':
                layer_name = 'features.denseblock4'
            elif self.cnf.train.backbone == 'shufflenet_v2_x2_0':
                layer_name = 'conv5'
            elif self.cnf.train.backbone == 'resnet18':
                layer_name = 'layer4'
            
            img_feature_object = FeatureExtractor(self.model1, layers = [layer_name])
            final_features = img_feature_object(x)[layer_name]
        # import IPython; IPython.embed()

        linear_input = einops.rearrange(self.avg_pool(final_features), 'b c 1 1 -> b c')
        print(f'linear_input {linear_input.shape}')
        linear_input = self.dropout(linear_input)
        logits = self.linear(linear_input)
        print(f'logits {logits.shape}')
        return logits
    
    def forward(self, x, iter_track, mode):
        if self.cnf.train.in_channels in (2,4):
            # apply feature fusion model 
            if self.cnf.train.in_channels == 2:
                im_ch_count = 1
            else:
                im_ch_count = 3

            image = x[:, 0:im_ch_count]
            mask = x[:, -1:]

            # img_feature_object = FeatureExtractor(self.model1, layers = ['conv5'])
            # msk_feature_object = FeatureExtractor(self.model2, layers = ['conv5'])

            f1_img = self.model1.forward1(image)
            f1_msk = self.model2.forward1(mask)
            

            gate_flags = [True]*len(self.cnf.train.fuse_direction)

            # # schedueler gated fusion applied on train time only else all gate on
            # if mode == 'train':
            #     count = len(self.cnf.train.fuse_direction) - self.cnf.train.fuse_direction.count('-')
            #     if self.cnf.train.gated_fusion and count != 0 :
            #         count = len(self.cnf.train.fuse_direction) - self.cnf.train.fuse_direction.count('-')
            #         gate_flags = [False]*len(self.cnf.train.fuse_direction)
            #         gate_flags[-1] = True
            #         for position in range(iter_track%count+1):
            #             gate_flags[len(self.cnf.train.fuse_direction)-1-position] = True

            # reverse gated fusion:
            # gate_flags = gate_flags[::-1]
            
            # random gated fusion
            # gate_flags = [True]*len(self.cnf.train.fuse_direction)
            # if self.cnf.train.gated_fusion:
            #     gate_flags[0] = random.choice([True, False])
            #     gate_flags[1] = random.choice([True, False])
            #     gate_flags[2] = random.choice([True, False])

            
            
            # **.** -------------------- conv1 fusion -------------------- **.**  
            if self.cnf.train.fuse_direction[0].upper() == 'M2I' and gate_flags[0]:
                f1_fused = self.fusion_conv1((f1_img + f1_msk)/2)
                f2_img = self.model1.forward2(f1_fused)
                f2_msk = self.model2.forward2(f1_msk)
            elif self.cnf.train.fuse_direction[0].upper() == 'I2M' and gate_flags[0]:
                f1_fused = self.fusion_conv1((f1_img + f1_msk)/2)
                f2_img = self.model1.forward2(f1_img)
                f2_msk = self.model2.forward2(f1_fused)
            elif self.cnf.train.fuse_direction[0].upper() == 'BD' and gate_flags[0]:
                f1_fused = (f1_img + f1_msk)/2
                
                f1_fused_im = self.fusion_conv1(f1_fused)
                f1_fused_msk = self.fusion_conv1_msk(f1_fused)
                
                f2_img = self.model1.forward2(f1_fused_im)
                f2_msk = self.model2.forward2(f1_fused_msk)
            else:
                f2_img = self.model1.forward2(f1_img)
                f2_msk = self.model2.forward2(f1_msk)
            
            # **.** -------------------- stage2 fusion -------------------- **.**  

            if self.cnf.train.fuse_direction[1].upper() == 'M2I' and gate_flags[1]:
                f2_fused = self.fusion_conv2((f2_img + f2_msk)/2)
                f3_img = self.model1.forward3(f2_fused)
                f3_msk = self.model2.forward3(f2_msk)
            elif self.cnf.train.fuse_direction[1].upper() == 'I2M' and gate_flags[1]:
                f2_fused = self.fusion_conv2((f2_img + f2_msk)/2)
                f3_img = self.model1.forward3(f2_img)
                f3_msk = self.model2.forward3(f2_fused)
            elif self.cnf.train.fuse_direction[1].upper() == 'BD' and gate_flags[1]:
                f2_fused = (f2_img + f2_msk)/2
                f2_fused_im = self.fusion_conv2(f2_fused)
                f2_fused_msk = self.fusion_conv2_msk(f2_fused)
                f3_img = self.model1.forward3(f2_fused_im)
                f3_msk = self.model2.forward3(f2_fused_msk)
            else:
                f3_img = self.model1.forward3(f2_img)
                f3_msk = self.model2.forward3(f2_msk)

            # **.** -------------------- stage3 fusion -------------------- **.**  

            if self.cnf.train.fuse_direction[2].upper() == 'M2I' and gate_flags[2]:
                f3_fused = self.fusion_conv3((f3_img + f3_msk)/2)
                f4_img = self.model1.forward4(f3_fused)
                f4_msk = self.model2.forward4(f3_msk)
            elif self.cnf.train.fuse_direction[2].upper() == 'I2M' and gate_flags[2]:
                f3_fused = self.fusion_conv3((f3_img + f3_msk)/2)
                f4_img = self.model1.forward4(f3_img)
                f4_msk = self.model2.forward4(f3_fused)
            elif self.cnf.train.fuse_direction[2].upper() == 'BD' and gate_flags[2]:

                f3_fused = (f3_img + f3_msk)/2
                
                f3_fused_im = self.fusion_conv3(f3_fused)
                f3_fused_msk = self.fusion_conv3_msk(f3_fused)
                
                
                f4_img = self.model1.forward4(f3_fused_im)
                f4_msk = self.model2.forward4(f3_fused_msk)
            else:
                f4_img = self.model1.forward4(f3_img)
                f4_msk = self.model2.forward4(f3_msk)

            # **.** -------------------- stage4 fusion -------------------- **.**  

            if self.cnf.train.fuse_direction[3].upper() == 'M2I' and gate_flags[3]:
                f4_fused = self.fusion_conv4((f4_img + f4_msk)/2)
                f5_img = self.model1.forward5(f4_fused)
                f5_msk = self.model2.forward5(f4_msk)
            elif self.cnf.train.fuse_direction[3].upper() == 'I2M' and gate_flags[3]:
                f4_fused = self.fusion_conv4((f4_img + f4_msk)/2)
                f5_img = self.model1.forward5(f4_img)
                f5_msk = self.model2.forward5(f4_fused)
            elif self.cnf.train.fuse_direction[3].upper() == 'BD' and gate_flags[3]:

                f4_fused = (f4_img + f4_msk)/2
                
                f4_fused_im = self.fusion_conv4(f4_fused)
                f4_fused_msk = self.fusion_conv4_msk(f4_fused)
                
                f5_img = self.model1.forward5(f4_fused_im)
                f5_msk = self.model2.forward5(f4_fused_msk)
            else:
                f5_img = self.model1.forward5(f4_img)
                f5_msk = self.model2.forward5(f4_msk)

            # **.** --------------- classifier level fusion -------------- **.**  
            if 'I' in self.cnf.train.fuse_direction[4].upper() and gate_flags[4]:
                final_features = f5_img
            elif 'M' in self.cnf.train.fuse_direction[4].upper() and gate_flags[4]:
                final_features = f5_msk
            else:
                final_features = f5_img + f5_msk

        else:
            if self.cnf.train.backbone == 'densenet121':
                layer_name = 'features.denseblock4'
            elif self.cnf.train.backbone == 'shufflenet_v2_x2_0':
                layer_name = 'conv5'
            elif 'resnet' in self.cnf.train.backbone:
                layer_name = 'layer4'
            
            img_feature_object = FeatureExtractor(self.model1, layers = [layer_name])
            final_features = img_feature_object(x)[layer_name]
        # import IPython; IPython.embed()
        linear_input = einops.rearrange(self.avg_pool(final_features), 'b c 1 1 -> b c')
        linear_input = self.dropout(linear_input)
        logits = self.linear(linear_input)
        return logits