import preprocessing
import pandas as pd
from PIL import Image

train_data = pd.read_csv('data/sign_mnist_train.csv')

y_train = train_data['label']
del train_data['label']


def preprocess_image(x):
    """
    we know that the pixel values lies between 0-255 but it is obsearved that models performs exceptionally well if we scale pixel values
    between 0-1"""
    x = x / 255
    x = x.reshape(-1, 28, 28)  # converting it into 28 x 28 gray scaled image

    return x


train_x = preprocess_image(train_data.values)
print(train_x[0].shape)

img = Image.fromarray(train_x[8], 'L')
img.show()
