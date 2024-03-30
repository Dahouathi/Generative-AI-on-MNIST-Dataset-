import numpy as np
from matplotlib import pyplot as plt


class RBM:
    def __init__(self, num_visible: int, num_hidden: int, mu: float = 0, sigma: float = 0.1):
        """ 
        Initialize a Restricted Boltzmann Machine (RBM) with the given dimensions and parameters.
        
        :param num_visible: The dimension of the visible nodes (V).
        :param num_hidden: The dimension of the hidden nodes (H).
        :param mu: The mean of the normal distribution used to initialize the weights. Default is 0.
        :param sigma: The standard deviation of the normal distribution used to initialize the weights. Default is 0.01.
        
        :return: A dictionary representing the RBM with initialized weights and biases.
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = np.random.normal(mu, sigma, size=(num_visible, num_hidden))
        self.a = np.random.randn(num_visible)
        self.b = np.random.randn(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def entree_sortie(self, v):
        """
        Compute the probability P(h=1|v) of the hidden nodes given the visible nodes.
        """
        return self.sigmoid(self.b + np.dot(v, self.W))
    
    def sortie_entree(self, h):
       
        """
        Compute the probability P(v=1|h) of the visible nodes given the hidden nodes.
        """
        return self.sigmoid(self.a + np.dot(h, self.W.T))

    def train(self, X, epochs=100, learning_rate=0.1, batch_size=128):
        """
        param X: training data
        param rbm: the RBM model
        param epochs: number of training epochs
        param learning_rate: learning rate
        param batch_size: size of mini-batches
        :return the trained RBM model
        """
        for epoch in range(epochs):
            x = X.copy()
            np.random.shuffle(x)
            mse = 0
            patience = 5 # Number of epochs to wait before early stopping
            wait = 0
            min_mse = np.inf  # Initialize minimum MSE to infinity

            for i in range(0, x.shape[0], batch_size):
                batch = x[i:i+batch_size]
                v0 = batch

                h0_prob = self.entree_sortie(v0)
                h0 = np.random.binomial(1, h0_prob)

                v1_prob = self.sortie_entree(h0)
                v1 = np.random.binomial(1, v1_prob)
                h1_prob = self.entree_sortie(v1)

                grad_W = np.dot(v0.T, h0) - np.dot(v1.T, h1_prob)
                grad_a = np.sum(v0 - v1, axis=0)
                grad_b = np.sum(h0 - h1_prob, axis=0)

                self.W += learning_rate * grad_W / batch_size
                self.a += learning_rate * grad_a / batch_size
                self.b += learning_rate * grad_b / batch_size

                mse += float(np.mean((v0 - v1) ** 2))
            mse /= x.shape[0] // batch_size
            print(f"Epoch {epoch + 1}/{epochs}, Mean Square Error : {mse}")
            # Check for improvement
            if mse < min_mse:
                min_mse = mse  # Update minimum MSE
                wait = 0  # Reset wait counter
            else:
                wait += 1  # Increment wait counter

            # Check for early stopping
            if wait >= patience:
                print("Early stopping due to no improvement in MSE.")
                break
            
            
    def generate_images(self, nb_images, x_shape=28, y_shape=28, nb_iterations=100, Plot=False):
        """
        param rbm: the trained RBM model
        param nb_iterations: number of iterations for Gibbs sampling
        param nb_images: number of images to generate
        """
        images = []
        for _ in range(nb_images):
            v = np.random.binomial(1, 0.5 * np.ones(self.num_visible))
            
            for _ in range(nb_iterations):
                h_prob = self.entree_sortie(v)
                h = np.random.binomial(1, h_prob)
                v_prob = self.sortie_entree(h)
                v = np.random.binomial(1, v_prob)
            
            images.append(v)
        
        if Plot:
            for i in range(nb_images):
                plt.subplot(1, nb_images, i+1)
                plt.imshow(images[i].reshape((x_shape, y_shape)), cmap='gray')  
                plt.axis('off')
            plt.show()

        return np.array(images)