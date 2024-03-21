import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))


def init_RBM(num_visible: int, num_hidden: int, mu: float = 0, sigma: float = 0.01):
    """ 
    Initialize a Restricted Boltzmann Machine (RBM) with the given dimensions and parameters.
    
    :param num_visible: The dimension of the visible nodes (V).
    :param num_hidden: The dimension of the hidden nodes (H).
    :param mu: The mean of the normal distribution used to initialize the weights. Default is 0.
    :param sigma: The standard deviation of the normal distribution used to initialize the weights. Default is 0.01.
    
    :return: A dictionary representing the RBM with initialized weights and biases.
    """
    # Initialize the weights with random values from a normal distribution
    W = np.random.normal(mu, sigma, size=(num_visible, num_hidden))
    
    # Initialize the biases with zeros
    a = np.zeros(num_visible) # bias for visible nodes
    b = np.zeros(num_hidden)  # bias for hidden nodes
    
    # Create the RBM structure with initialized weights and biases
    rbm = {'W': W, 'a': a, 'b': b}
    
    return rbm



def entree_sortie_rbm(rbm, v):
        return sigmoid(rbm['b'] + np.dot(v, rbm['W']))


def sortie_entree_rbm(rbm, h):
        return sigmoid(rbm['a'] + np.dot(h, rbm['W'].T))


def train_RBM(X, rbm, epochs=100, learning_rate=0.1, batch_size=128):
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

        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i+batch_size]
            v0 = batch

            
            h0_prob = entree_sortie_rbm(rbm, v0)
            h0 = np.random.binomial(1, h0_prob)

            
            v1_prob = sortie_entree_rbm(rbm, h0)
            v1 = np.random.binomial(1, v1_prob)
            h1_prob = entree_sortie_rbm(rbm, v1)

            grad_W = np.dot(v0.T, h0) - np.dot(v1.T, h1_prob)
            grad_a = np.sum(v0 - v1, axis=0)
            grad_b = np.sum(h0 - h1_prob, axis=0)

            # Mise à jour des poids et biais
            rbm['W'] += learning_rate * grad_W / batch_size
            rbm['a'] += learning_rate * grad_a / batch_size
            rbm['b'] += learning_rate * grad_b / batch_size

            mse += np.mean((v0 - v1) ** 2)

        mse /= x.shape[0] // batch_size
        print(f"Epoch {epoch + 1}/{epochs}, Mean Square Error : {mse}")

    return rbm

def generer_image_RBM(rbm, nb_images,x_shape=28, y_shape=28, nb_iterations=100, Plot=False ):
    """
    param rbm: the trained RBM model
    param nb_iterations: number of iterations for Gibbs sampling
    param nb_images: number of images to generate
    """
    # Initialize the visible units randomly
    p = rbm['W'].shape[0]
    images = []
    for i in range(nb_images):
        v = np.random.binomial(1, 0.5 * np.ones(p))
        
        for _ in range(nb_iterations):
            
            h_prob = entree_sortie_rbm(rbm, v)
            h = np.random.binomial(1, h_prob)
            v_prob = sortie_entree_rbm(rbm, h)
            v = np.random.binomial(1, v_prob)
        images.append(v)
    if Plot:
        for i in range(nb_images):
            plt.subplot(1, nb_images, i+1)
            plt.imshow(images[i].reshape((x_shape, y_shape)), cmap='gray')  
            plt.axis('off')
        plt.show()

    return np.array(images)


