from ..dbn import init_DBN, train_DBN
from ..rbm import entree_sortie_rbm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import pandas as pd

def init_DNN(sizes, output_size=10):
    """
    Initialize a Deep Neural Network (DNN) with the given sizes and output size.
    param: sizes: list of integers representing the number of nodes in each layer
    param: output_size: number of nodes in the output layer
    return: the initialized DNN model
    """
    configuration = sizes + [output_size]
    print(configuration)
    return init_DBN(configuration)

def pretrain_DNN(X, dnn, epochs=100, learning_rate=0.1, batch_size=128, verbose=False):
    """
    Pretrain a Deep Neural Network (DNN) using the given data.
    param: X: training data
    param: dnn: the DNN model
    param: epochs: number of training epochs
    param: learning_rate: learning rate
    param: batch_size: size of mini-batches
    return: the pretrained DNN model
    """
    dbn = dnn[:-1]
    dbn = train_DBN(X, dbn, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, verbose=verbose)
    dnn[:-1] = dbn
    return dnn


def calcul_softmax(rbm, X):
    """
    Calculate the softmax probabilities for the given layer and input data.
    param: layer: a dictionary containing the weights 'W' and biases 'b' of the layer
    param: X: input data
    return: softmax probabilities
    """
    # Calculate the logits
    logits = np.dot(X, rbm['W']) + rbm['b']
    
    # Apply the softmax function
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return softmax_probs

def entree_sortie_reseau(DNN, X):
    """
    Compute the output of the Deep Neural Network (DNN) given the input.
    param: DNN: the DNN model
    param: X: the input data
    return: the output of the DNN
    """
    v = X.copy()
    sorties = [X]  # List to store the outputs of each layer
    proba_sortie = None  # Will store the output of the last layer
    
    for i in range(len(DNN)-1):
        rbm = DNN[i]
        p_h = entree_sortie_rbm(rbm, v)
        v = np.random.binomial(1, p_h)
        sorties.append(p_h)
    rbm_classification = DNN[-1]
    proba_sortie = calcul_softmax(rbm_classification, v)
    
    return sorties, proba_sortie

def retropropagation(X, y, dnn, epochs=100, learning_rate=0.1, batch_size=128, verbose=True, plot=True, pretrained=False, save_path=None):
    """
    Train a Deep Neural Network (DNN) using the given data.
    param: X: training data
    param: y: target labels
    param: dnn: the DNN model
    param: pretrained: whether the DNN is pretrained
    param: epochs: number of training epochs
    param: learning_rate: learning rate
    param: batch_size: size of mini-batches
    param: verbose: whether to print the loss at each epoch
    param: plot: whether to plot the loss
    return: the trained DNN model"""
    # Convert y to one-hot encoded format using pd.get_dummies
    y_one_hot = pd.get_dummies(y).values
    previous_loss=100
    best_loss = float('inf')  # Initialize the minimum loss to infinity
    loss = []
    patience = 7  # Number of epochs to wait before early stopping
    wait = 0
    
    for epoch in range(epochs):
        # Shuffle the data using sklearn shuffle
        X_shuffled, y_shuffled = shuffle(X, y_one_hot)
        
        loss_batches = []
        # Iterate over each mini-batch
        for batch in range(0, X.shape[0], batch_size):
            
            X_batch = X_shuffled[batch: min(batch + batch_size, X.shape[0]), :]
            y_batch = y_shuffled[batch: min(batch + batch_size, X.shape[0])]

            # Forward propagation
            sorties, proba_sortie = entree_sortie_reseau(dnn, X_batch)
            

            # Loss calculation
            loss_batch = -np.mean(np.sum(y_batch * np.log(proba_sortie), axis=1))
            loss_batches.append(loss_batch)
            
            # Backward propagation
            # Last layer
            delta = proba_sortie - y_batch
            grad_W = np.dot(sorties[-1].T, delta) / batch_size
            grad_b = np.mean(delta, axis=0)
            dnn[-1]['W'] -= learning_rate * grad_W
            dnn[-1]['b'] -= learning_rate * grad_b

            # Hidden layers
            for i in range(2, len(dnn)+1):
                if i == 2:  # Classification layer
                    RBM = dnn[-1]
                    delta = np.dot(delta, RBM['W'].T) * sorties[-1] * (1 - sorties[-1])
                else:
                    RBM = dnn[-i+1]
                    delta = np.dot(delta, RBM['W'].T) * sorties[-i+1] * (1 - sorties[-i+1])
                if i == len(dnn):
                    grad_W = np.dot(X_batch.T, delta)
                else:
                    grad_W = np.dot(sorties[-i].T, delta)
                
                grad_b = np.mean(delta, axis=0)
    
                # Update weights and biases
                dnn[-i]['W'] -= learning_rate * grad_W
                dnn[-i]['b'] -= learning_rate * grad_b


        # Calculate the cross entropy loss for the epoch
        
        train_loss = float(np.mean(loss_batches))
        loss.append(train_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}")

        # Early stopping strategy
        if epoch>epochs//4:
            # Check if current loss is less than the best loss encountered so far
            if train_loss < best_loss:
                # If so, update the best loss and reset wait
                if abs(previous_loss - train_loss) < 1e-3: # if the loss is not decreasing
                    wait+=1
                else:
                    wait = 0
                best_loss = train_loss
                # reset wait since we've seen improvement
            else:
                wait += 1  # increment wait since there was no improvement

        # If we have waited for 'patience' epochs without improvement, stop training
        if wait >= patience:
            print("Early stopping due to no improvement in Loss.")
            break
        previous_loss=train_loss
          
    
    if plot:
        plt.figure()
        plt.plot(np.arange(len(loss)), loss)
        plt.xlabel("Epochs")
        plt.ylabel("CrossEntropy Loss")
        if pretrained:
            plt.title("Loss for pretrained DNN")
        else:
            plt.title("Loss for DNN (without pretraining)")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.close()
    return dnn


def test_dnn(X, y, dnn, verbose=False):
    _, proba_sortie = entree_sortie_reseau(dnn, X)
    predictions = np.argmax(proba_sortie, axis=1)
    accuracy = np.mean(predictions == y)
    if verbose :
        print(f"Accuracy: {accuracy}")
    return accuracy


def box_plot_proba(data, dnn, k, save_path=None):
    _, proba_sortie = entree_sortie_reseau(dnn, data)
    
    # Take the first 20 samples
    proba_sortie = proba_sortie[:20]
    
    fig, axs = plt.subplots(5, 4, figsize=(15, 15))
    axs = axs.ravel()
    for i, probas in enumerate(proba_sortie):
        axs[i].plot(range(10), probas)
        axs[i].set_title(f'Sample {i+1}')
        axs[i].set_xlabel('Classes')
        axs[i].set_ylabel('Predicted probability')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()
    