from mrcnn.allobj import AllobjConfig
import tensorflow as tf
import mrcnn.model as modellib
import numpy as np

config = AllobjConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class FindObjects:
  def __init__(self):

    self.config = InferenceConfig()
    DEVICE = "/gpu:0"
    WEIGHTS_PATH = "networks/obj_detection/mask_rcnn_1.h5"

    with tf.device(DEVICE):
      self.model = modellib.MaskRCNN(mode="inference",model_dir='',config=self.config)
    self.model.load_weights(WEIGHTS_PATH, by_name=True)

    self.dict2names = {
      1 : 'moon',
      2 : 'tree'
    }


  def __call__(self, image):
    image = np.asarray(image)
    results = self.model.detect([image], verbose=1)[0]

    all_classes = [self.dict2names[cur_class] for cur_class in results['class_ids']]
    if len(all_classes) > 0:
      masks = np.split(results['masks'], axis=2, indices_or_sections=range(1, len(all_classes)))
      all_masks = [mask.squeeze() for mask in masks]
    else:
      all_masks = []
    
    
    return all_classes, all_masks