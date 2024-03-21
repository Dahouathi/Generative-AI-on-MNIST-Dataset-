from ..rbm.RBM_class import RBM
from matplotlib import pyplot as plt
import numpy as np
class DBN:
    def __init__(self, sizes):
        self.rbms = [RBM(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    def train(self, X, epochs=100, learning_rate=0.1, batch_size=128):
        # Initializations
        n_layers = len(self.rbms)
        X_train = X.copy()
        # training layer by layer
        for i in range(n_layers):
            print('Training layer', i+1)
            self.rbms[i].train(X_train, epochs, learning_rate, batch_size)
            X_train = np.random.binomial(1, self.rbms[i].entree_sortie(X_train))

    def generate_images(self, nb_images, x_shape=28, y_shape=28, nb_iterations=100, Plot=False):
        """
        Generate images using the Deep Belief Network (DBN).
        """
        v = self.rbms[-1].generate_images(nb_images, x_shape, y_shape, nb_iterations, False)
        for i in range(2, len(self.rbms)+1):
            v = np.random.binomial(1, self.rbms[-i].sortie_entree(v))
        images = v.reshape((nb_images, x_shape, y_shape))

        if Plot:
            # Reshape and Plot generated images
            fig, axes = plt.subplots(1, nb_images, figsize=(10, 2))
            for i in range(nb_images):
                axes[i].imshow(images[i], cmap="gray")
                axes[i].axis("off")
            plt.suptitle(f"Number of layers {len(self.rbms)}")
            plt.show()
        return images