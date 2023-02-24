import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from PIL import Image


train_path = r"D:\PhD_Program\Deep-Learning\Datasets\CIFAR-10\train"
test_path = r"D:\PhD_Program\Deep-Learning\Datasets\CIFAR-10\test"
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def show_image(image_array):
    fig = plt.figure(0)
    fig.set_size_inches(5, 5)
    for i in range(0, 32):
        fig.add_subplot(8, 8, i+1)
        img = np.array(np.clip(image_array[i], 0, 255), dtype=np.uint8).reshape(32,32,3)
        plt.imshow(img)    
    plt.show()

def load_data(data_path):
    features = []
    lables = []

    for cl in classes:
        path = os.path.join(data_path, cl)
        cls_num = classes.index(cl)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img)) 
            # BGR -> RGB
            img_array = img_array[:,:,::-1]
            # print(img_array.shape)  
            features.append(img_array)
            lables.append(cls_num)
            
    return np.asarray(features, dtype=np.uint8), np.asarray(lables) 


train_features, train_labels = load_data(train_path)
test_features, test_labels = load_data(test_path)

print("Training lenght: {} and Testing lenght: {}".format(len(train_features), len(test_features)))

X_train = train_features.reshape(50000, 32*32*3)
X_test = test_features.reshape(10000, 32*32*3)
y_train = train_labels.flatten()
y_test = test_labels.flatten()

print("Training features shape: {} lables: {}".format(X_train.shape, y_train.shape))
print("Testing features shape: {} lables: {}".format(X_test.shape, y_test.shape))

# show_image(X_test)

# def rgb2gray_array(rgb):
    
#     r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

#     return gray

# train_features_gray = rgb2gray_array(train_features)

# fig = plt.figure(0)
# fig.set_size_inches(18.5, 18.5)
# for i in range(0,32):
#     fig.add_subplot(8, 8, i+1)
#     plt.imshow(train_features_gray[i], cmap='gray')

# plt.show()

from sklearn.decomposition import PCA
import pickle

pca = PCA(n_components=400)
pca.fit(X_train, y_train)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.ylim(0.8, 1.0)
# plt.grid()
# plt.show()


pca = PCA(n_components=2)
pca.fit(train_features)
scaled_train_features = pca.transform(train_features)
# save pca in a pickle file
with open('pca.pkl', 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)
# For the other script where you want to use the fitted pca -

with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)
scaled_data = pca.transform(data) 




