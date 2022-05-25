import preprocessing
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# train_data = pd.read_csv('data/sign_mnist_train.csv')

# y_train = train_data['label']
# del train_data['label']


# def preprocess_image(x):
#     """
#     we know that the pixel values lies between 0-255 but it is obsearved that models performs exceptionally well if we scale pixel values
#     between 0-1"""
#     x = x / 255
#     x = x.reshape(-1, 28, 28)  # converting it into 28 x 28 gray scaled image

#     return x


# train_x = preprocess_image(train_data.values)
# print(train_x[0].shape)

# img = Image.fromarray(train_x[8], 'L')
# img.show()


trainloader, _ = preprocessing.get_train_test_loader()
print(trainloader)


it = iter(trainloader)
train = next(it)

train_features = train['image']
train_labels = train['label']











# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

fig, ax = plt.subplots(2, 5)

k = 0
for i in range(2) :
    for j in range(5):

        img = train_features[k].squeeze()
        label = train_labels[k]
        ax[i, j].imshow(img, cmap="gray")
        ax[i,j].set_title(str(label))
        print(f"Label: {label}")
        k = k + 1

plt.show()
