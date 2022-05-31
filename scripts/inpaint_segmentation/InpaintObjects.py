import numpy as np
import pandas as pd
import random
from collections import Counter


class InpaintObjects:

    def __init__(self):
        NORMAL_CLASSES = {93, 96, 105, 106, 109, 110, 111, 118, 119, 120, 123, 124, 125, 126, 127,128, 129, 133, 134, 135, 141, 142, 147, 148, 149, 153, 154, 156, 158, 159, 161, 162, 168, 177, 178, 179, 181}
        ALL_CLASSES = {i for i in range(182)}
        self.GARBAGE_CLASSES = ALL_CLASSES.difference(NORMAL_CLASSES)
        self.TREE_CLASS = 168
        self.SKY_CLASS = 156

    def fill_more_segmemtation(self, x, y, shift, segmentation):
        if (x >= 0 and y>=0 and x<len(segmentation) and y<len(segmentation[0])):
            segmentation[x, y] = np.nan
        if shift > 0:
            self.fill_more_segmemtation(x-1, y, shift-1, segmentation)
            self.fill_more_segmemtation(x+1, y, shift-1, segmentation)
            self.fill_more_segmemtation(x, y-1, shift-1, segmentation)
            self.fill_more_segmemtation(x, y+1, shift-1, segmentation)

    def fill_more(self, new_inpaint_mask):
        old_mask = new_inpaint_mask.copy()
        i = 0
        for x in range(1, len(old_mask)-1):
            for y in range(1, len(old_mask[0])-1):
                if old_mask[x, y] == 1 and (old_mask[x-1, y] != 1 or old_mask[x+1, y] != 1 or old_mask[x, y-1] != 1 or old_mask[x, y+1] != 1):
                    i += 1
                    new_inpaint_mask[x, y] = 0
        return new_inpaint_mask

    def inpaint_segmentation(self, inpaint_mask, segmentation, new_classes, new_masks):
        segmentation, add_forest = self.change_forest_segm(segmentation)
        SHIFT = int(max(len(inpaint_mask), len(inpaint_mask[0])) / 50)
        new_inpaint_mask = inpaint_mask.copy()
        for _ in range(SHIFT):
            new_inpaint_mask = self.fill_more(new_inpaint_mask)
        new_segmentation = segmentation.copy()
        new_segmentation[new_inpaint_mask==0] = np.nan

        new_inpaint_segmentation = new_segmentation.copy()
        top_x, top_y, bottom_x, bottom_y = None, None, None, None
        for x in range(0, len(new_segmentation)):

            #get last point
            last_p = new_segmentation[x, 0]

            if add_forest and last_p == self.TREE_CLASS and top_x is None:
                top_x, top_y = x, 0

            inds = [0, int(len(new_segmentation[0])/3), int(2*len(new_segmentation[0])/3), len(new_segmentation[0])-1]
            for i in inds:
                if new_segmentation[x, i] == self.TREE_CLASS or new_segmentation[x, i] in self.GARBAGE_CLASSES:
                    last_p = self.SKY_CLASS
                else:
                    last_p = new_segmentation[x, i]
                    break

            for y in range(1, len(new_segmentation[0])):
                if add_forest:
                    if new_segmentation[x,y] == self.TREE_CLASS:
                        if not top_x:
                            top_x = x
                        if not top_y or y < top_y:
                            top_y = y
                        if not bottom_x or x > bottom_x:
                            bottom_x = x
                        if not bottom_y or y > bottom_y:
                            bottom_y = y

                #change current point
                if (new_inpaint_mask[x,y] == 0 or new_segmentation[x,y] in self.GARBAGE_CLASSES):
                    new_inpaint_segmentation[x,y] = self.get_most_common([last_p, last_p,
                                                                          new_inpaint_segmentation[x - 1,y] if x > 0 else last_p,
                                                                          new_inpaint_segmentation[x,y + 1] if y < len(new_segmentation[0]) - 1 else last_p,
                                                                          new_inpaint_segmentation[x + 1,y] if x < len(new_segmentation) - 1 else last_p
                                                                          ])

                # update last point
                if new_inpaint_segmentation[x,y] != self.TREE_CLASS:
                    last_p = new_inpaint_segmentation[x,y]

        if add_forest:
            new_classes, new_masks = self.add_forest_masks(top_x, top_y, bottom_x, bottom_y, new_classes, new_masks)
        return new_segmentation, new_inpaint_segmentation, new_classes, new_masks

    def is_beach(self, segmentation):
        beach = {153, 149}
        stats = Counter(np.array(segmentation).flatten()).most_common()[:4]
        for pixel_stat in stats[:4]:
            if pixel_stat[0] in beach:
                return True
        return False

    def get_most_common_in_mask(self, segmentation, mask):
        data = np.ma.masked_array(segmentation, ~mask)
        return self.get_most_common(data[data.mask == False].data)

    def get_most_common(self, points):
        return Counter(points).most_common()[0][0]

    def add_forest_masks(self, top_x, top_y, bottom_x, bottom_y, new_classes, new_masks):
        h, full_w = bottom_x - top_x, bottom_y - top_y
        k_tree = 1.8
        w = int(h / k_tree)
        shift = w // 2
        tree_num = int(full_w / shift)
        if tree_num < 5:
            return new_classes, new_masks

        top, left, bottom, right = top_x, top_y, bottom_x, top_x + w
        while left < bottom_y:
            new_classes.append('tree')
            new_masks.append([top, left, bottom, right])
            y_shift = random.randint(10, 30)
            if top == top_x:
                top -= y_shift
                bottom -= y_shift
            else:
                top = top_x
                bottom = bottom_x
            left = left + shift
            right = right + shift
        return new_classes, new_masks

    def check_forest_type(self, segm):
        h, w = len(segm), len(segm[0])
        stats = Counter(np.array(segm).flatten()).most_common()[:4]
        from_top = (segm[0][0] == self.TREE_CLASS and
                    segm[0][w // 2] == self.TREE_CLASS and
                    segm[0][w - 1] == self.TREE_CLASS)
        tree_cnt = 0
        for stat in stats:
            if stat[0] == self.TREE_CLASS:
                tree_cnt = stat[1]
                break
        tree_percent = tree_cnt / (h * w)
        if from_top and stats[0][0] == self.TREE_CLASS and tree_percent > 0.55:
            return 'large_forest'
        if tree_percent > 0.25:
            return 'middle_forest'
        return '-'

    def change_large_forest(self, segm):
        h, w = len(segm), len(segm[0])
        h1, h2, h3 = h // 5, h // 3, h // 2
        w1, w2, w3 = w // 3, w // 5, w // 8
        segm[:h1,w1:2*w1][segm[:h1,w1:2*w1] == self.TREE_CLASS] = self.SKY_CLASS
        segm[h1:h2,w2:4*w2][segm[h1:h2,w2:4*w2] == self.TREE_CLASS] = self.SKY_CLASS
        segm[h2:h3,w3:7*w3][segm[h2:h3,w3:7*w3] == self.TREE_CLASS] = self.SKY_CLASS
        return segm

    def change_forest_segm(self, segm):
        forest_type = self.check_forest_type(segm)
        print("TYPE:", forest_type)
        if forest_type == 'middle_forest':
            return segm, True
        elif forest_type == 'large_forest':
            return self.change_large_forest(segm), False
        return segm, False

    def __call__(self, classes, masks, segmentation):
        segmentation = segmentation.numpy().astype(float)
        new_classes, new_masks = [], []
        if len(masks) == 0:
            new_segmentation, new_inpaint_segmentation, new_classes, new_masks = self.inpaint_segmentation(np.ones_like(segmentation), segmentation, new_classes, new_masks)
            return new_classes, new_masks, new_segmentation, new_inpaint_segmentation, new_segmentation, new_inpaint_segmentation
        inpaint_mask = np.ones_like(masks[0])

        is_beach = self.is_beach(segmentation)
        for i in range(len(masks)):
            if not is_beach and (classes[i] != 'tree' or self.get_most_common_in_mask(segmentation, masks[i]) == self.TREE_CLASS):
                new_classes.append(classes[i])
                new_masks.append(masks[i])

                inpaint_mask[masks[i]] = 0

        segmentation[inpaint_mask==0] = np.nan

        new_segmentation, new_inpaint_segmentation, new_classes, new_masks = self.inpaint_segmentation(inpaint_mask, segmentation, new_classes, new_masks)

        df = pd.DataFrame(segmentation)
        inpaint_segmentation = df.interpolate(method='pad', axis=1).to_numpy()

        return new_classes, new_masks, segmentation, inpaint_segmentation, new_segmentation, new_inpaint_segmentation
