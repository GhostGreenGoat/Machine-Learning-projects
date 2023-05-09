import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pca(X, D):
    """
    PCA
    Input:
        X - An (1024, 1024) numpy array
        D - An int less than 1024. The target dimension
    Output:
        X - A compressed (1024, 1024) numpy array
    """
    #do SVD to X
    U,S,V= np.linalg.svd(X, full_matrices=False)
    #select the top D principal components
    U_reduced = U[:, :D]
    S_reduced = S[:D]
    V_reduced = V[:D, :]
    #Reconstruct X
    X_reduced = np.dot(U_reduced * S_reduced, V_reduced)
    return X_reduced

    
    raise NotImplementedError


def sklearn_pca(X, D):
    """
    Your PCA implementation should be equivalent to this function.
    Do not use this function in your implementation!
    """
    from sklearn.decomposition import PCA
    p = PCA(n_components=D, svd_solver='full')
    trans_pca = p.fit_transform(X)
    X = p.inverse_transform(trans_pca)
    return X


    

if __name__ == '__main__':
    D = 256

    a = Image.open('data/20180108_171224.jpg').convert('RGB')
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Original')
    ax1.imshow(a)
    b = np.array(a)
    c = b.astype('float') / 255.
    for i in range(3):
        x = c[:, :, i]
        mu = np.mean(x)
        x = x - mu
        x_true = sklearn_pca(x, D)
        x = pca(x, D)
        assert np.allclose(x, x_true, atol=0.05)  # Test your results
        x = x + mu
        c[:, :, i] = x

    b = np.uint8(c * 255.)
    a = Image.fromarray(b)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Compressed')
    ax2.imshow(a)
    plt.show()
