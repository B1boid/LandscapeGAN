from PIL import Image


class GetReference:

    def __init__(self, is_mock=True):
        self.is_mock = is_mock
        self.mock_path = 'images/input_examples/'
        self.mock_names = [
         'tree_1.jpg',
         'tree_moon_1.jpeg',
         'field_1.jpg',
         'trees_1.jpg',
         'moon_1.jpg',
         'trees_2.jpg'
        ]

    def photostock_request(self, text):
        return None

    def __call__(self, input_file=0, text=None):
        if self.is_mock:
            file_name = self.mock_names[input_file] if isinstance(input_file, int) else input_file
            return Image.open(self.mock_path + file_name)
        return self.photostock_request(text)

