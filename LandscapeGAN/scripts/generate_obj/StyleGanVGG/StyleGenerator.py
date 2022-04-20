import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import numpy as np
from stylegan2.model import Generator
from StyleLoss import VGGStyleLoss
from torch import optim



class StyleGenerator:
    def __init__(self, network_pkl, steps=30):
        self.device= torch.device("cuda")
        self.steps = steps
        self.g_ema = Generator(1024, 512, 8)
        self.g_ema.load_state_dict(torch.load(network_pkl)["g_ema"], strict=False)
        self.g_ema.eval()
        self.g_ema = self.g_ema.cuda()
        self.lr = 0.1
        self.l2_lambda = 0.001
        self.vgg_loss = VGGStyleLoss(0)
        
    def image_loader(self, image, is_path):
        if is_path:
            image=Image.open(image)
        new_sz = 1024
        loader=transforms.Compose([transforms.Resize((new_sz,new_sz)), transforms.ToTensor()])
        image=loader(image).unsqueeze(0)
        return image.to(self.device,torch.float)
    
    def generate_image_from_latent(self):
        #mean_latent = self.g_ema.mean_latent(4096)
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            img_orig, latent_code_init = self.g_ema([latent_code_init_not_trunc], return_latents=True)
        return img_orig, latent_code_init
    
    def generate_images_random_walk(self,ref_path, style_image_path, num):
        if style_image_path:
          style_image = self.image_loader(style_image_path, True)
        if ref_path:
          ref_image = self.image_loader(ref_path, True)
        l1_criterion = nn.L1Loss()
        best_images = []
        for i in range(self.steps):
            img_gen, _ = self.generate_image_from_latent()
            with torch.no_grad(): 
              
              if style_image_path and ref_path:
                c_loss = self.vgg_loss(img_gen, style_image)
                l1_loss = l1_criterion(img_gen, ref_image)
                loss = l1_loss + 0.03 * c_loss
              elif style_image_path:
                c_loss = self.vgg_loss(img_gen, style_image)
                loss = c_loss
              elif ref_path:
                l1_loss = l1_criterion(img_gen, ref_image)
                loss = l1_loss

            #print(f"loss: {loss.item():.4f} c_loss: {(0.03*c_loss.item()):.4f} l1_loss: {l1_loss.item():.4f}")
            #print(f"loss: {loss.item():.4f} l1_loss: {l1_loss.item():.4f}")
            best_images.append((img_gen, loss))
        
        sorted_imgs = sorted(best_images, key=lambda x: x[1])
        # if num == 1:
        #     best_img = sorted_imgs[0]
        #     #print("Best loss:", best_img[1])
        #     return ToPILImage()(make_grid(best_img[0].detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))

        res = []
        for img in sorted_imgs:
            r_img = ToPILImage()(make_grid(img[0].detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
            if r_img not in res:
                res.append(r_img)
                #print("Best loss:", img[1])
            if len(res) == num:
              break
       
        return res
    
    def generate_images_latent_walk(self, ref_path, style_image_path, num):
        style_image = self.image_loader(style_image_path, True)
        ref_image = self.image_loader(ref_path, True)
        #vgg_loss = VGGLoss(style_image, False)
        _, latent_code_init = self.generate_image_from_latent()
        
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

        optimizer = optim.Adam([latent], lr=self.lr)
        l1_criterion = nn.L1Loss()
        best_images = []
        for i in range(self.steps):
            img_gen, _ = self.g_ema([latent], input_is_latent=True, randomize_noise=False)
            
            #c_loss = vgg_loss(img_gen)
            c_loss = self.vgg_loss(img_gen, style_image)
            l1_loss = l1_criterion(img_gen, ref_image)
            loss = l1_loss + 0.01 * c_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item():.4f} c_loss: {c_loss.item():.4f} l1_loss: {l1_loss.item():.4f}")
        
        return ToPILImage()(make_grid(img_gen.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
        
    def generate_style_images(self, ref_path, style_image_path, mode='LatentWalk', num=1):
        if mode == 'LatentWalk':
            return self.generate_images_latent_walk(ref_path, style_image_path, 1)
        return self.generate_images_random_walk(ref_path, style_image_path, num)