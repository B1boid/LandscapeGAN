import requests
import random
from PIL import Image


class RequestImage:

    def __init__(self):
        self.config = {
            "pixabay": {
                "url": 'https://pixabay.com/api/?key=23887201-f5feaa19f925d9487e48c07d1&q={}&image_type=photo&per_page=30&page={}',
                "tag_results": 'hits',
                "tag_image": {
                    'medium': ['webformatURL'],
                    'large': ['largeImageURL']
                }
            },
            "unsplash": {
                "url": 'https://api.unsplash.com/search/photos?query={}&client_id=kt4I9LIgyzPpqPxmqkMYnTzqlTPmUNFQLJxjIahksd0&per_page=30&page={}',
                "tag_results": 'results',
                "tag_image": {
                    'medium': ['urls', 'small'],
                    'large': ['urls', 'regular']
                }
            }
        }

    def get_batch(self, class_name, config, img_type, limit=60):
        data = []
        page = 1
        while True:
            if len(data) >= limit:
                break
            try:
                URL = config['url'].format(class_name, page)
                response = requests.get(URL)
                json = response.json()
                if config['tag_results'] != "":
                    json = json[config['tag_results']]
                for el in json:
                    for tag in config['tag_image'][img_type]:
                        el = el[tag]
                    data.append(el)
            except:
                break
        return data[:limit]

    def get_random_img_url(self, class_name, img_type='medium'):
        platform_mode = random.randint(0, 1)
        if platform_mode % 2 == 0:
            batch = self.get_batch(class_name, self.config['pixabay'], img_type)
            if len(batch) == 0:
                batch = self.get_batch(class_name, self.config['unsplash'], img_type)
        else:
            batch = self.get_batch(class_name, self.config['unsplash'], img_type)
            if len(batch) == 0:
                batch = self.get_batch(class_name, self.config['pixabay'], img_type)

        if len(batch) == 0:
            raise Exception('Nothing found with these tags')

        ind = random.randint(0, len(batch) - 1)
        return batch[ind]


    def __call__(self, tags):
        url = self.get_random_img_url(tags)
        return Image.open(requests.get(url, stream=True).raw)
