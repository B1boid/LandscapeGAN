import warnings
warnings.filterwarnings("ignore")
import sys

sys.path.append('scripts/obj_detection')
sys.path.append("scripts/generate_obj/StyleGanVGG")
sys.path.append("scripts/generate_obj")
sys.path.append("scripts/combine_images")
sys.path.append("tt")

from scripts.reference_input.GetReference import GetReference
from scripts.segmentation_input.GetSegmentation import GetSegmentation
from scripts.obj_detection.FindObjects import FindObjects
from scripts.inpaint_segmentation.InpaintObjects import InpaintObjects
from scripts.generate_obj.GenerateObjects import GenerateObjects
from scripts.generate_obj.GenerateRainbow import GenerateRainbow
from scripts.clear_obj.ClearObjects import ClearObjects
from scripts.combine_images.CombineImages import CombineImages
from scripts.request_img.RequestImage import RequestImage
from tt.GetBackgroundImage import GetBackgroundImage
from scripts.utils.base64 import base64_to_pil, pil_to_base64
import numpy as np


class LandscapeGan:

    def __init__(self):
        self.getReference = GetReference()
        self.getSegmentation = GetSegmentation()
        self.findObjects = FindObjects()
        self.inpaintObjects = InpaintObjects()
        self.getBackgroundImage = GetBackgroundImage()
        self.generateRainbow = GenerateRainbow()
        self.generateObjects = GenerateObjects()
        self.clearObjects = ClearObjects()
        self.combineImages = CombineImages()
        self.requestImage = RequestImage()
        print("Successful init LandscapeGan")

    def generate_from_img(self, reference_image, need_tt, is_img_base64=True):
        result = {}
        if not reference_image:
            raise Exception("`Reference image` argument is None")
        if is_img_base64:
            input_image = base64_to_pil(reference_image)
        else:
            input_image = reference_image
        input_segmentation = self.getSegmentation(input_image)
        all_classes, all_masks = self.findObjects(input_image)
        all_classes, all_masks, _, _, _, new_inpaint_segm = self.inpaintObjects(all_classes, all_masks, input_segmentation)

        tt_image_arr, tt_image, _ = self.getBackgroundImage(new_inpaint_segm)

        if need_tt:
            if np.array_equal(np.array(new_inpaint_segm).astype(int), np.array(input_segmentation).astype(int)):
                original_tt_image = tt_image.copy()
            else:
                _, original_tt_image, _ = self.getBackgroundImage(input_segmentation)
            result['tt_image'] = pil_to_base64(original_tt_image)
            result['input_image'] = pil_to_base64(input_image)

        tt_image = self.generateRainbow(tt_image, tt_image_arr, new_inpaint_segm, False)

        objs_dict = self.generateObjects(all_classes, tt_image)
        clean_objs_dict = self.clearObjects(objs_dict)

        resultImage = self.combineImages(all_classes, all_masks, new_inpaint_segm, tt_image, clean_objs_dict)

        result['result'] = pil_to_base64(resultImage)
        return result

    def generate_from_tags(self, tags, need_tt):
        if not tags:
            raise Exception("`Tags` argument is None")
        input_image = self.requestImage(tags)
        return self.generate_from_img(input_image, need_tt, is_img_base64=False)

    def generate(self, mode='img', **params):
        if mode == 'tags':
            return self.generate_from_tags(params.get(mode), params.get('need_tt'))
        return self.generate_from_img(params.get(mode), params.get('need_tt'))