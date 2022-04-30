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
from tt.GetBackgroundImage import GetBackgroundImage
from scripts.utils.base64 import base64_to_pil, pil_to_base64


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
        print("Successful init LandscapeGan")

    def generate_from_img(self, reference_image):
        if not reference_image:
            raise Exception("`Reference image` argument is None")

        try:
            input_image = base64_to_pil(reference_image)
        except Exception as e:
            return {'input_image_error': str(e)}

        try:
            input_segmentation = self.getSegmentation(input_image)
        except Exception as e:
            return {'getSegmentation_error': str(e)}

        all_classes, all_masks = self.findObjects(input_image)
        try:
            all_classes, all_masks, _, _, _, new_inpaint_segm = self.inpaintObjects(all_classes, all_masks, input_segmentation)
        except Exception as e:
            return {'inpaintObjects_error': str(e)}

        try:
            tt_image_arr, tt_image, _ = self.getBackgroundImage(new_inpaint_segm)
        except Exception as e:
            return {'getBackgroundImage_error': str(e)}
        try:
            tt_image = self.generateRainbow(tt_image, tt_image_arr, new_inpaint_segm, False)
        except Exception as e:
            return {'generateRainbow_error': str(e)}

        try:
            objs_dict = self.generateObjects(all_classes, tt_image)
        except Exception as e:
            return {'generateObjects_error': str(e)}
        try:
            clean_objs_dict = self.clearObjects(objs_dict)
        except Exception as e:
            return {'clearObjects_error': str(e)}

        try:
            resultImage = self.combineImages(all_classes, all_masks, new_inpaint_segm, tt_image, clean_objs_dict)
        except Exception as e:
            return {'combineImages_error': str(e)}

        try:
            res = {"result": pil_to_base64(resultImage)}
        except Exception as e:
            return {'pil_to_base64_error': str(e)}

        return res

    def generate_from_tags(self, tags):
        if not tags:
            raise Exception("`Tags` argument is None")
        return 'not implemented'

    def generate(self, mode='img', **params):
        if mode == 'tags':
            return self.generate_from_tags(params.get(mode))
        return self.generate_from_img(params.get(mode))
