import scipy as sp
import numpy as np

def lire_mnist(data_path: str, indices: np.ndarray, data_type: str, thrshold: int = 127):
    """
    :param data_path: path to import data
    :param indices: index of alpha digits we want to use for training
    :param data_type: "train", "test"
    :param thrshold: threshold to binarize the images
    :return: array (n, p): n number of sample, p number of pixels
    """
    mnist_all = sp.io.loadmat(data_path, simplify_cells=True)

    data_mnist = []
    label = []
    for i in indices:
        key = data_type + str(i)
        data_mnist.append((mnist_all[key] > thrshold).astype(int))
        label.append(i * np.ones(mnist_all[key].shape[0]))

    data_mnist = np.vstack(data_mnist)
    label = np.concatenate(label, axis=0).astype(int)
    return data_mnist, label