from numpy.core import arrayprint
from omegaconf import OmegaConf
import yaml
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.append('scripts/segmentation_input')

from extract_segmentation import COCOStuffSegmenter, make_segm_tensor
from sflckr import SegmentationBase



class GetSegmentation:

    def make_segmentation_tensor(self, img):
        segm_template = Image.open('images/templates/segmentation.png')

        class Examples(SegmentationBase):
            def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
                super().__init__([img], segm_template, size=size, random_crop=random_crop, interpolation=interpolation)
        model_seg = COCOStuffSegmenter({}).cuda()

        dataset = Examples()
        dloader = DataLoader(dataset, batch_size=1)
        return make_segm_tensor(dataloader=dloader,model=model_seg).squeeze()

    def __call__(self, img):
        return self.make_segmentation_tensor(img)



