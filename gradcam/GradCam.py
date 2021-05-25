"""
Inspired on:
https://www.kaggle.com/chanhu/residual-attention-network-pytorch
"""

import numpy as np
import cv2
import os
 
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from utils.utils import imsave, imshow

class GradCam:
    def __init__(self, model, cam_layer_name, imagenet_norm=False):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.cam_layer_name = cam_layer_name
        self.imagenet_norm = imagenet_norm

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        feature_maps = []
        
        for i in range(x.size(0)):
            img = x[i].data.cpu().numpy()
            img = img.transpose((1, 2, 0))
            # apply imagenet de-normalization
            if self.imagenet_norm:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

            feature = x[i].unsqueeze(0)
            
            for name, module in self.model.named_children():
                if isinstance(module, torch.nn.Linear): #name == self.last_fc:
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == self.cam_layer_name:
                    feature.register_hook(self.save_gradient)
                    self.feature = feature

            classes = torch.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            classes.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
                
            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = feature_map * 0.4 + np.float32((np.uint8(img * 255)))
            cam = cam - np.min(cam)
            
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
                
            feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
            
        feature_maps = torch.stack(feature_maps)
        
        return feature_maps


def computeGradCam(gradcam, test_loader, device, RESULTS_DIR, model_name):
    print('*' * 50)
    print('GradCAM started:')
    all_feature_img = []
    for x_batch, y_batch in test_loader:
        for x, y in zip(x_batch, y_batch):
            img_tensor = x.unsqueeze(dim=0).to(device)
            feature_img = gradcam(img_tensor).squeeze(dim=0)
            all_feature_img.append(feature_img)

    out = torchvision.utils.make_grid(all_feature_img, nrow=8, padding=1)
    gradcam_figname = '%s/gradcam_%s.pdf' % (RESULTS_DIR, model_name)
    imshow(out, figsize=(20, 40), filename=gradcam_figname, imagenet=False)

    print('GradCAM saving images:')
    gradcam_figpath = '%s/gradcam_imgs_%s' % (RESULTS_DIR, model_name)
    try:
        os.makedirs(gradcam_figpath, exist_ok=True)
        print("Directory '%s' created successfully" % gradcam_figpath)
    except OSError as error:
        print("Directory '%s' can not be created or allready exist" % gradcam_figpath)

    imsave(all_feature_img, gradcam_figpath, test_loader.dataset.samples, imagenet=False)