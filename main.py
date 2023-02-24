import numpy as np 
import pandas as pd
from pca import PCA
from zca import ZCA

data_path = 'data\iris.csv'

df = pd.read_csv(data_path)

'''
Let your (centered) data be stored in a [n x d] matrix X with d features (variables) in columns and n data points in rows. 
'''

features = df.drop(columns=['species']).values
lables = df['species'].values

print("The size of the dataset: {}".format(len(features)))
# print(df.head())

'''
Take 3 examples for testing

X_1 = [6, 3, 4.8, 1.8, virginica]
X_2 = [6.7, 3.1, 4.7, 1.5, versicolor]
X_3 = [5, 3.5, 1.6, 0.6, setosa]

'''


def main():
    '''
    save eigenvector if needed
    '''    
    pca_model = PCA(n_components=2, whiten=True)
    pca_model.fit(features)

    # eigenvectors = pca_model.eigvect
    outfile = "pca_eigenvector.npz"
    # np.savez(outfile, eigen_vectors = eigenvectors)
    
    
    eigen_data = np.load(outfile)
    eigenvectors = eigen_data['eigen_vectors']    
    
    x_test = np.array([[6.7, 3.1, 4.7, 1.5]])

    print("X:", x_test)   
    print("Mean X: {}".format(pca_model.mean_))
    print("Standard deviation X: {}".format(pca_model.stdv_))

    x_standardized = (x_test - pca_model.mean_) / pca_model.stdv_

    print("Nomarlized X:", x_standardized)

    reduced_data = x_standardized @ eigenvectors
    print("Pre-trained PCA:", reduced_data)
   
    
    # print(pca_model.variance_ratio)
    reduced_new = pca_model.transform(x_test)
    print("Online PCA:", reduced_new)   

    zca_model = ZCA()
    zca_model.fit(features)

    whitening = zca_model.transform(x_test)
    print("ZCA whitened:", whitening)

    print("[INFO] PCA testing ....")

    # Generate pca weights for traning dataste
    weights = pca_model.transform(features)
    print("Shape of the weight matrix:", weights.shape)
    euclidean_distance = np.linalg.norm(weights - reduced_new, axis=1)
    best_match = np.argmin(euclidean_distance)
    print("PCA best match '%s' with Euclidean: %f at Index: %d" % (lables[best_match], euclidean_distance[best_match], best_match))

    print("[INFO] ZCA testing ....")
    # Generate zca weights for traning dataste
    weights = zca_model.transform(features)
    print("Shape of the weight matrix:", weights.shape)
    euclidean_distance = np.linalg.norm(weights - whitening, axis=1)
    best_match = np.argmin(euclidean_distance)
    print("ZCA best match '%s' with Euclidean: %f at Index: %d" % (lables[best_match], euclidean_distance[best_match], best_match))



if __name__ == '__main__':
    main()
