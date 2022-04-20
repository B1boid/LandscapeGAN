import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import cv2
import math
from PIL import Image, ImageDraw, ImageFilter
import random

class OverlayImages:

    def img_float32(self, img):
        return img.copy() if img.dtype != 'uint8' else (img/255.).astype('float32')
    
    def over(self,fgimg, bgimg):
        fgimg, bgimg = self.img_float32(fgimg),self.img_float32(bgimg)
        (fb,fg,fr,fa),(bb,bg,br,ba) = cv2.split(fgimg),cv2.split(bgimg)
        color_fg, color_bg = cv2.merge((fb,fg,fr)), cv2.merge((bb,bg,br))
        alpha_fg, alpha_bg = np.expand_dims(fa, axis=-1), np.expand_dims(ba, axis=-1)
        
        color_fg[fa==0]=[0,0,0]
        color_bg[ba==0]=[0,0,0]
        
        a = fa + ba * (1-fa)
        a[a==0]=np.NaN
        color_over = (color_fg * alpha_fg + color_bg * alpha_bg * (1-alpha_fg)) / np.expand_dims(a, axis=-1)
        color_over = np.clip(color_over,0,1)
        color_over[a==0] = [0,0,0]
        
        result_float32 = np.append(color_over, np.expand_dims(a, axis=-1), axis = -1)
        return (result_float32*255).astype('uint8')
    
    def overlay_with_transparency(self, bgimg, fgimg, xmin = 0, ymin = 0,trans_percent = 0.5):
        '''
        bgimg: a 4 channel image, use as background
        fgimg: a 4 channel image, use as foreground
        xmin, ymin: a corrdinate in bgimg. from where the fgimg will be put
        trans_percent: transparency of fgimg. [0.0,1.0]
        '''
        #we assume all the input image has 4 channels
        assert(bgimg.shape[-1] == 4 and fgimg.shape[-1] == 4)
        fgimg = fgimg.copy()
        roi = bgimg[ymin:ymin+fgimg.shape[0], xmin:xmin+fgimg.shape[1]].copy()
        
        b,g,r,a = cv2.split(fgimg)
        
        fgimg = cv2.merge((b,g,r,(a*trans_percent).astype(fgimg.dtype)))
        
        roi_over = self.over(fgimg,roi)
        
        result = bgimg.copy()
        result[ymin:ymin+fgimg.shape[0], xmin:xmin+fgimg.shape[1]] = roi_over
        return result

    def expand_dim(self, img_arr):
        i = img_arr.copy()
        #red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
        if i.shape[-1] == 3:
          i = np.insert(i, 3, 255, axis=2)
        return i; 

    def __call__(self, background_img_arr, foreground_img):
        bck = self.expand_dim(background_img_arr)
        obj = np.array(foreground_img)
        res = self.overlay_with_transparency(bck, obj)
        return Image.fromarray(res)


class GenerateRainbow:

  def __init__(self):
    self.SKY = {105, 106, 156}

  def find_sky_size(self, segmentation):
    last_h = 0
    w, h = len(segmentation[0]), len(segmentation)
    for hh in range(0, h, 5):
      if segmentation[hh, 0] in self.SKY or segmentation[hh, 100] in self.SKY:
        last_h = hh
      else:
        break
    return w, last_h

  def get_white_mask(self, im):
    data = np.array(im)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if data[x][y][3] > 0:
                data[x][y] = np.array([255, 255, 255, 255])

    return Image.fromarray(data)

  def get_rainbow_mask(self, img_back):
      bck_black = Image.new('RGBA', (img_back.size[0], img_back.size[1]), (0, 0, 0, 255))
      tmp_white = self.get_white_mask(img_back)
      bck_black.paste(tmp_white, (0, 0), tmp_white)
      return bck_black


  def get_rainbow(self, sky_size, img_size):
      w, h = sky_size
      img = Image.new("RGBA", sky_size)
      img1 = ImageDraw.Draw(img)
      MODE = random.randint(0, 2)
      if MODE == 0:
        WIDTH = int (w / 400)
        shape = [(-2*w, -h), (w + w/20, 4*h)]
        paste_coords = (30, 0)
      elif MODE == 1:
        WIDTH = int (w / 600)
        shape = [(-w/10, -h/10), (2*w, 2*h)]
        paste_coords = (-10, 0)
      elif MODE == 2:
        WIDTH = int (w / 300)
        shape = [(-w/10, -h/4), (2*w, 2*h)] 
        paste_coords = (-10, 0) 
      # elif MODE == 3:
      #   WIDTH = int (w / 400)
      #   shape = [(-w, -h/4), (1.1 * w, 1.8*h)]
      #   paste_coords = (10, 0)
      WIDTH = max(WIDTH, 1)
      print(WIDTH)
      colors = [(255,0 , 0), (255, 20, 0), (255, 50, 0),  (255, 75, 0), (255, 100, 0), (255, 150, 0),(255, 200, 0), (242, 255, 0),(174, 255, 0), (217,255,0),
                (106, 255, 0), (38, 255, 0),(0, 255, 94),(0, 255, 162),
                (0, 255, 230), (0, 217, 255), (0, 149, 255), (0, 81, 255), (0, 13, 255),
                (55, 0, 255), (119, 0, 255), (187, 0, 255), (255, 0, 255)
                #(255, 128, 0), (255, 255, 0), (0, 255,0), (0, 255, 255), (0,0,255), (128, 0, 255)
                ]
      colors = sum([[color]*WIDTH for color in colors], [])

      for i, color in enumerate(colors):
        alpha = 50
        if len(colors) - i < WIDTH * 1.5 or i < WIDTH * 1.5:
          alpha = 15
        elif len(colors) - i < WIDTH * 4.5 or i < WIDTH * 4.5:
          alpha = 30
        img1.ellipse(shape, outline=(*color, alpha))
        if MODE == 0:
          shape = [(shape[0][0]+1, shape[0][1]+1),(shape[1][0]-1, shape[1][1]-1)]
        elif MODE == 1:
          shape = [(shape[0][0]+1, shape[0][1]+1),(shape[1][0]+1, shape[1][1]+1)]
        elif MODE == 2:
          shape = [(shape[0][0], shape[0][1]+1),(shape[1][0], shape[1][1]+1)]

      img_back = Image.new("RGBA", img_size)
      img_back.paste(img,paste_coords)
      if random.randint(0,2) != 1:
        img_back = img_back.transpose(Image.FLIP_LEFT_RIGHT)

      return img_back

  def __call__(self, img, img_arr, segmentation, need_rainbow):
    if not need_rainbow:
      return img

    img = img.convert("RGBA")
    rainbow_img = self.get_rainbow(self.find_sky_size(segmentation),img.size)
    rainbow_img = rainbow_img.filter(ImageFilter.BLUR)
    rainbow_img = rainbow_img.filter(ImageFilter.SMOOTH_MORE)
    rainbow_img = rainbow_img.filter(ImageFilter.BLUR)
    
    overlayImages = OverlayImages()
    res = overlayImages(img_arr, rainbow_img)
   
    return res
