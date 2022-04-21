from numpy.core import arrayprint
from omegaconf import OmegaConf
import yaml
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader


from taming.models.cond_transformer import Net2NetTransformer


class GetBackgroundImage:

    def __init__(self):
        config_path = "tt/logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
        self.config = OmegaConf.load(config_path)
        self.model = Net2NetTransformer(**self.config.model.params)
        ckpt_path = "tt/logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(sd, strict=False)
        self.model.cuda().eval()
        torch.set_grad_enabled(False)

    def img_from_segmentation(self, s):
        s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
        colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
        colorize = colorize / colorize.sum(axis=2, keepdims=True)
        s = s@colorize
        s = s[...,0,:]
        s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
        return Image.fromarray(s)

    def process_segmentation(self, segmentation):
        segmentation = np.array(segmentation).astype(int)
        segmentation = np.eye(182)[segmentation]
        segmentation = segmentation.transpose(2,0,1)
        return torch.tensor(segmentation[None]).to(dtype=torch.float32)
       
    def encode_image(self, segmentation):   
        c_code, c_indices = self.model.encode_to_c(segmentation.to(self.model.device))
        return c_code, c_indices

    def decode_image(self, c_code):
        return self.model.cond_stage_model.decode(c_code)

    def return_image(self, s):
        s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
        s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
        return s, Image.fromarray(s)

    def generate_image(self, c_code, c_indices):
        codebook_size = self.config.model.params.first_stage_config.params.embed_dim
        z_indices_shape = c_indices.shape
        z_code_shape = c_code.shape
        z_indices = torch.randint(codebook_size, z_indices_shape, device=self.model.device)
        x_sample = self.model.decode_to_img(z_indices, z_code_shape)

        idx = z_indices
        idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])

        cidx = c_indices
        cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

        temperature = 1.0
        top_k = 100
        update_every = 50

        for i in range(0, z_code_shape[2]-0):
          if i <= 8:
            local_i = i
          elif z_code_shape[2]-i < 8:
            local_i = 16-(z_code_shape[2]-i)
          else:
            local_i = 8
          for j in range(0,z_code_shape[3]-0):
            if j <= 8:
              local_j = j
            elif z_code_shape[3]-j < 8:
              local_j = 16-(z_code_shape[3]-j)
            else:
              local_j = 8

            i_start = i-local_i
            i_end = i_start+16
            j_start = j-local_j
            j_end = j_start+16
            
            patch = idx[:,i_start:i_end,j_start:j_end]
            patch = patch.reshape(patch.shape[0],-1)
            cpatch = cidx[:, i_start:i_end, j_start:j_end]
            cpatch = cpatch.reshape(cpatch.shape[0], -1)
            patch = torch.cat((cpatch, patch), dim=1)
            logits,_ = self.model.transformer(patch[:,:-1])
            logits = logits[:, -256:, :]
            logits = logits.reshape(z_code_shape[0],16,16,-1)
            logits = logits[:,local_i,local_j,:]

            logits = logits/temperature

            if top_k is not None:
              logits = self.model.top_k_logits(logits, top_k)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx[:,i,j] = torch.multinomial(probs, num_samples=1)

            step = i*z_code_shape[3]+j
            if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
              x_sample = self.model.decode_to_img(idx, z_code_shape)
            
        return x_sample
        
    def __call__(self, segm_img_arr):
        torch.cuda.empty_cache()
        segm_tensor = self.process_segmentation(segm_img_arr)
        c_code, c_indices = self.encode_image(segm_tensor)
        generated_img = self.generate_image(c_code, c_indices)
        generated_img_arr, generated_img = self.return_image(generated_img)
        return generated_img_arr,generated_img, self.img_from_segmentation(segm_tensor)

