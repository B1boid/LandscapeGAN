import warnings
warnings.filterwarnings("ignore")
import sys

sys.path.append('scripts/obj_detection')
sys.path.append("scripts/generate_obj/StyleGanVGG")
sys.path.append("scripts/generate_obj")
sys.path.append("scripts/combine_images")

from scripts.reference_input.GetReference import GetReference
from scripts.segmentation_input.GetSegmentation import GetSegmentation
#from scripts.obj_detection.FindObjects import FindObjects
from scripts.inpaint_segmentation.InpaintObjects import InpaintObjects
#from scripts.generate_obj.GenerateObjects import GenerateObjects
from scripts.generate_obj.GenerateRainbow import GenerateRainbow
from scripts.clear_obj.ClearObjects import ClearObjects
from scripts.combine_images.CombineImages import CombineImages


class LandscapeGan:

    def __init__(self):
        self.getReference = GetReference()
        self.getSegmentation = GetSegmentation()
        #self.findObjects = FindObjects()
        self.inpaintObjects = InpaintObjects()
        self.generateRainbow = GenerateRainbow()
        #self.generateObjects = GenerateObjects()
        self.clearObjects = ClearObjects()
        self.combineImages = CombineImages()
        print("Successful init LandscapeGan")

    def generate_from_img(self, reference_image):
        if not reference_image:
            raise Exception("`Reference image` argument is None")
        return 'not implemented'

    def generate_from_tags(self, tags):
        if not tags:
            raise Exception("`Tags` argument is None")
        return 'not implemented'

    def generate(self, mode='img', **params):
        if mode == 'tags':
            return self.generate_from_tags(params.get(mode))
        return self.generate_from_img(params.get(mode))
