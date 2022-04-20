from StyleGanVGG.StyleGenerator import StyleGenerator
import numpy as np


class GenerateObjects:

  def __init__(self):
    path = 'networks/stylegan/'
    self.models_dict = {
      'moon': StyleGenerator(path + 'moon.pt'),
      'tree': StyleGenerator(path + 'tree.pt')
    }
    self.ref_img_path = 'images/templates/tree.jpg'
    self.st_img_path = 'images/templates/tree1.jpg'
    self.moon_img_path = 'images/templates/moon.png'

  def need_green(self, background_img):
    w, h = background_img.size
    cur_img = background_img.crop((0, h//2, w, h))
    npimg = np.array(cur_img)
    average = npimg.mean(axis=0).mean(axis=0)
    r,g,b = average[0],average[1],average[2]
    if r / g > 1.3 or np.mean([r,g,b]) > 160:
      return False
    return True

  def get_objs_dict(self, arr):
    counts = dict()
    for obj in arr:
      counts[obj] = counts.get(obj, 0) + 1
    return counts

  def __call__(self, all_classes, style_img=None):
    obj_dict = self.get_objs_dict(all_classes)
    if not(style_img and self.need_green(style_img)):
      print("classic")
      self.st_img_path = None
    results = {}
    for obj_key in obj_dict:
      if obj_key == 'tree':
        results[obj_key] = self.models_dict[obj_key].generate_style_images(self.ref_img_path, self.st_img_path, mode='RandomWalk', num=obj_dict[obj_key])
      elif obj_key == 'moon':
        results[obj_key] = self.models_dict[obj_key].generate_style_images(None, self.moon_img_path, mode='RandomWalk', num=obj_dict[obj_key])
    return results