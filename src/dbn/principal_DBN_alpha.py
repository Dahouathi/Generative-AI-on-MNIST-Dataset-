from ..rbm.principal_RBM_alpha import init_RBM
from ..rbm.principal_RBM_alpha import entree_sortie_rbm
from ..rbm.principal_RBM_alpha import sortie_entree_rbm 
from ..rbm.principal_RBM_alpha import train_RBM
from ..rbm.principal_RBM_alpha import generer_image_RBM
import numpy as np
from matplotlib import pyplot as plt


def init_DBN(sizes):
    """
    Initialize a Deep Belief Network (DBN) with the given sizes
    
    param:
    sizes: a list containing the number of visible units and hidden units for each RBM in the DBN
    
    Returns:
    DBN: a list containing the weights and biases of each RBM in the DBN
    """
    dbn = []  # List to store each RBM
    
    # Initialization of each RBM in the DBN
    for i in range(len(sizes) - 1):
        rbm = init_RBM(sizes[i], sizes[i+1])
        dbn.append(rbm)
    
    return dbn

def train_DBN(X, dbn, epochs=100, learning_rate=0.1, batch_size=128, verbose=False):
    """
    Train a Deep Belief Network (DBN) using the Contrastive Divergence (CD) algorithm
    
    param:
    dbn: a list containing the weights and biases of each RBM in the DBN
    X: a 2D numpy array containing the input data
    n_iter: number of iterations for the training
    batch_size: size of the mini-batches
    learning_rate: learning rate for the training
    
    Returns:
    dbn: a list containing the weights and biases of each RBM in the DBN
    """
    # Initialization of variables
    n_layers = len(dbn)
    X_train = X.copy()
    
    # Layer by layer training
    for i in range(n_layers):
        print('Training layer', i+1)
        rbm = dbn[i]  # current RBM
        train_RBM(X_train, rbm, epochs, learning_rate, batch_size, verbose)
        X_train = np.random.binomial(1, entree_sortie_rbm(rbm, X_train))
        dbn[i] = rbm
    
    return dbn

def generate_image_DBN(dbn, nb_images, x_shape=28, y_shape=28, nb_iterations=100, Plot=True, save_path=None):
    """
    Generates images using a Deep Belief Network (DBN).

    param:
    - dbn (list): The Deep Belief Network containing the weights and biases.
    - nb_images (int): The number of images to generate.
    - x_shape (int): The width of the generated images (default: 28).
    - y_shape (int): The height of the generated images (default: 28).
    - nb_iterations (int): The number of iterations for generating each image (default: 100).
    - Plot (bool): Whether to plot the generated images (default: False).

    Returns:
    - images (ndarray): An array of generated images with shape (nb_images, x_shape, y_shape).
    """
    rbm = dbn[-1]
    v = generer_image_RBM(rbm, nb_images, x_shape, y_shape, nb_iterations, False)
    
    for i in range(2, len(dbn)+1):
        rbm = dbn[-i]
        v = np.random.binomial(1, sortie_entree_rbm(rbm, v))

    images = v.reshape((nb_images, x_shape, y_shape))
    if Plot:
        # Reshape and Plot generated images
        fig, axes = plt.subplots(1, nb_images, figsize=(10, 2))
        for i in range(nb_images):
            axes[i].imshow(images[i])
            axes[i].axis("off")
        plt.suptitle(f"Number of layers {len(dbn)}")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show(block=False)
    return images