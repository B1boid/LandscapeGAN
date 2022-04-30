from mrcnn.allobj import AllobjConfig
import tensorflow as tf
import mrcnn.model as modellib
import numpy as np
from keras.backend import clear_session

config = AllobjConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class FindObjects:
  def __init__(self):

    print("Log-1")
    self.config = InferenceConfig()
    DEVICE = "/gpu:0"
    WEIGHTS_PATH = "networks/obj_detection/mask_rcnn_1.h5"
    print("Log-2")
    with tf.device(DEVICE):
      self.model = modellib.MaskRCNN(mode="inference",model_dir='',config=self.config)
    print("Log-3")
    self.model.load_weights(WEIGHTS_PATH, by_name=True)
    print("Log-4")
    self.model.keras_model._make_predict_function()
    print("Log-5")

    self.graph = tf.get_default_graph()

    self.dict2names = {
      1 : 'moon',
      2 : 'tree'
    }
    print("Log-5-")


  def __call__(self, image):
    print("Log-6")
    image = np.asarray(image)
    print("Log-7")
    with self.graph.as_default():
      results = self.model.detect([image], verbose=1)[0]
    print("Log-8")
    clear_session()
    print("Log-9")

    all_classes = [self.dict2names[cur_class] for cur_class in results['class_ids']]
    if len(all_classes) > 0:
      masks = np.split(results['masks'], axis=2, indices_or_sections=range(1, len(all_classes)))
      all_masks = [mask.squeeze() for mask in masks]
    else:
      all_masks = []
    print("Log-10")
    
    return all_classes, all_masks