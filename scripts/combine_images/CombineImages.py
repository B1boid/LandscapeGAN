from PIL import Image
import numpy as np
from PostprocessImage import PostprocessImage



class CombineImages:


    def corners(self, np_array):
        result = np.where(np_array == np.amax(np_array))
        top = np.min(result[0])
        bottom = np.max(result[0])
        left = np.min(result[1])
        right = np.max(result[1])
        return top, left, bottom, right

    def get_new_size(self, cur_obj, corners_vals):
        top, left, bottom, right = corners_vals
        old_w, old_h = cur_obj.size
        ref_h = bottom - top
        ref_w = right - left
        k_w = ref_w / old_w
        k_h = ref_h / old_h
        k = np.average([k_w, k_h])
        return int(old_w * k), int(old_h * k)

    def get_new_coords(self, new_w, new_h, corners_vals, shift):
        top, left, bottom, right = corners_vals
        coord_x = left + ((right - left) // 2) - (new_w // 2)
        coord_y = bottom - new_h + shift

        return coord_x, coord_y


    def __call__(self, all_classes, all_masks, segmentation, tt_image, clean_objs_dict):
        bck = tt_image.convert("RGBA")
        ind_dict = {}
        for i, obj_key in enumerate(all_classes):
            shift = 0
            if obj_key == 'tree':
                shift = 25
            ind_dict[obj_key] = ind_dict.get(obj_key, -1) + 1
            corners_vals = self.corners(all_masks[i])
            cur_obj = clean_objs_dict[obj_key][ind_dict[obj_key]]
            cur_obj = cur_obj.crop(cur_obj.getbbox())

            new_w, new_h = self.get_new_size(cur_obj, corners_vals)
            coord_x, coord_y = self.get_new_coords(new_w, new_h, corners_vals, shift)

            cur_obj = cur_obj.resize((new_w, new_h), Image.ANTIALIAS)
            bck.paste(cur_obj, (coord_x, coord_y), cur_obj)
        postprocessImage = PostprocessImage(all_classes, segmentation)
        return postprocessImage(bck)
