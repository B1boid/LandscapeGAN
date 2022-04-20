import numpy as np
import pilgram

from collections import Counter
from catboost import CatBoostClassifier

from skimage.color import rgb2gray, rgba2rgb
from sklearn.cluster import KMeans
import functools
import operator

class ImageFeatures:

    def dtype_limits(self, image, clip_negative=False):
        _integer_types = (np.byte, np.ubyte,          # 8 bits
                          np.short, np.ushort,        # 16 bits
                          np.intc, np.uintc,          # 16 or 32 or 64 bits
                          int, np.int_, np.uint,      # 32 or 64 bits
                          np.longlong, np.ulonglong)  # 64 bits
        _integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                           for t in _integer_types}
        dtype_range = {bool: (False, True),
                       np.bool_: (False, True),
                       np.bool8: (False, True),
                       float: (-1, 1),
                       np.float_: (-1, 1),
                       np.float16: (-1, 1),
                       np.float32: (-1, 1),
                       np.float64: (-1, 1)}
        dtype_range.update(_integer_ranges)
        imin, imax = dtype_range[image.dtype.type]
        if clip_negative:
            imin = 0
        return imin, imax

    def get_brightness(self, image):
        greyscale_image = image.convert('L')
        histogram = greyscale_image.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale

    def get_contrast(self, image, fraction_threshold=0.05, lower_percentile=1,
                     upper_percentile=99, method='linear'):
        image = np.asanyarray(image)

        if image.dtype == bool:
            return not ((image.max() == 1) and (image.min() == 0))

        if image.ndim == 3:

            if image.shape[2] == 4:
                image = rgba2rgb(image)
            if image.shape[2] == 3:
                image = rgb2gray(image)

        dlimits = self.dtype_limits(image, clip_negative=False)
        limits = np.percentile(image, [lower_percentile, upper_percentile])
        ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])

        return ratio

    def get_average_color(self, image):
        w, h = image.size
        img = np.array(image.resize((w//4, h//4)))
        img = img[:, :, 0:3]
        return list(map((lambda x: -np.log2(x / 255)), np.average(img, axis=(0,1))))

    def get_main_colors(self, image):
        clt = KMeans(3)
        w, h = image.size

        clt.fit(np.array(image.resize((w//4, h//4)))[:, :, 0:3].reshape(-1, 3))
        n_pixels = len(clt.labels_)
        counter = Counter(clt.labels_)
        perc = {}
        res = []
        for i in counter:
            res.append((clt.cluster_centers_[i], np.round(counter[i]/n_pixels, 2)))
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [el[0] for el in res]
        return list(map((lambda x: x / 255), functools.reduce(operator.iconcat, res, [])))

    def get_features(self, img, has_beach=False):
        return [int(has_beach), self.get_brightness(img), self.get_contrast(img), *self.get_average_color(img), *self.get_main_colors(img)]


class PostprocessImage:

    def __init__(self, all_classes=None, segmentation=None):
        self.is_night = 'moon' in all_classes
        self.need_filter = 'tree' in all_classes
        self.is_beach = self.is_beach(segmentation)

    def is_beach(self, segmentation):
        beach = {153, 149}
        stats = Counter(segmentation.flatten()).most_common()[:4]
        for pixel_stat in stats[:4]:
            if pixel_stat[0] in beach:
                return True
        return False

    def dark_filter(self, image):
        return pilgram.mayfair(image)

    def light_filter(self, image):
        return pilgram.clarendon(pilgram.gingham(image))

    def __call__(self, image):
        print(self.is_night, self.is_beach, self.need_filter)
        if self.is_night:
            return self.dark_filter(image)
        if self.is_beach:
            return pilgram.gingham(image)

        model = CatBoostClassifier()
        model.load_model('networks/postprocess/cat_model')

        filters_dict = {
            0: lambda x: x,
            1: self.light_filter,
            2: self.dark_filter
        }

        pred_probas = model.predict_proba(ImageFeatures().get_features(image))
        print(pred_probas)

        if self.need_filter:
            if pred_probas[1] > pred_probas[2]:
                return self.light_filter(image)
            else:
                return self.dark_filter(image)
        return filters_dict[np.argmax(pred_probas)](image)