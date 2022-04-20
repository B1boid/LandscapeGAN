from PIL import Image
import numpy as np

class ClearObjects:
    
    def __init__(self):
        self.method = {
            "moon": self.delete_green,
            "tree": self.delete_white_blue
        }
    
    def delete_green(self, im, bound=130, k=1.4):
        data = np.array(im)

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                if data[x][y][1] > bound and data[x][y][1] > k * data[x][y][0] and data[x][y][1] > k * data[x][y][2]:
                    data[x][y] = np.array([0, 0, 0, 0])

        return Image.fromarray(data)

    def delete_white_blue(self, im):
        data = np.array(im)

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                if data[x][y][0] > 50 and data[x][y][1] > 30 and data[x][y][2] > 110:
                    data[x][y] = np.array([0, 0, 0, 0])

        return Image.fromarray(data)

    def __call__(self, objs_dict):
        results = {}
        for obj_key in objs_dict:
          results[obj_key] = []
          for obj_img in objs_dict[obj_key]:
            results[obj_key].append(self.method[obj_key](obj_img.convert('RGBA')))
        return results