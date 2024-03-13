import scipy as sp
import numpy as np

def lire_alpha_digit(data_path: str, indices=None):
    """
    :param data_path: path to import data
    :param indices: index of alpha digits we want to use for training
    :return: array (n, p): n number of sample, p number of pixels
    """
    data = sp.io.loadmat(data_path, simplify_cells=True)
    alpha_digits = data['dat']
    
    if indices is not None:
        alpha_digits = alpha_digits[indices]
    
    images = np.zeros((alpha_digits.size, alpha_digits[0, 0].size))
    im = 0  # image index
    for i in range(alpha_digits.shape[0]):
        for j in range(alpha_digits.shape[1]):
            images[im, :] = alpha_digits[i, j].flatten()
            im += 1

    return images