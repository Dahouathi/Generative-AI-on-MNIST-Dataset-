from ..dbn import init_DBN, train_DBN
from ..rbm import entree_sortie_rbm
import numpy as np
from matplotlib import pyplot as plt


def init_DNN(sizes, output_size=10):
    """
    Initialize a Deep Neural Network (DNN) with the given sizes and output size.
    param: sizes: list of integers representing the number of nodes in each layer
    param: output_size: number of nodes in the output layer
    return: the initialized DNN model
    """
    configuration = sizes + [output_size]
    return init_DBN(configuration)

def pretrain_DNN(X, dnn, epochs=100, learning_rate=0.1, batch_size=128):
    """
    Pretrain a Deep Neural Network (DNN) using the given data.
    param: X: training data
    param: dnn: the DNN model
    param: epochs: number of training epochs
    param: learning_rate: learning rate
    param: batch_size: size of mini-batches
    return: the pretrained DNN model
    """
    dbn= {'W': dnn['W'][:-1], 'a': dnn['a'][:-1], 'b': dnn['b'][:-1]}
    dbn = train_DBN(X, dbn, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
    dnn['W'][:-1] = dbn['W']
    dnn['a'][:-1] = dbn['a']
    dnn['b'][:-1] = dbn['b']
    return dnn


def calcul_softmax(rbm, X):
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
    sorties = [X]  # List to store the outputs of each layer
    proba_sortie = None  # Will store the output of the last layer
    
    for i in range(len(DNN['W'])-1):
        RBM = {'W': DNN['W'][i], 'a': DNN['a'][i], 'b': DNN['b'][i]}
        v = np.random.binomial(1, entree_sortie_rbm(RBM, sorties[-1]))
        sorties.append(v)
    RBM = {'W': DNN['W'][-1], 'a': DNN['a'][-1], 'b': DNN['b'][-1]}
    proba_sortie = calcul_softmax(RBM, sorties[-1])
    
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
    min_loss = np.inf  # Initialize the minimum loss to infinity
    loss = []
    patience = 5
    wait = 0
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Split the data into mini-batches
        num_batches = len(X) // batch_size
        X_batches = np.array_split(X_shuffled, num_batches)
        y_batches = np.array_split(y_shuffled, num_batches)
        
        loss_batches = []
        # Iterate over each mini-batch
        for X_batch, y_batch in zip(X_batches, y_batches):

            # Forward propagation
            sorties, proba_sortie = entree_sortie_reseau(dnn, X_batch)
            
            # Convert y_batch to one-hot encoded format
            num_classes = proba_sortie.shape[1]
            y_one_hot = np.eye(num_classes)[y_batch]

            # Loss calculation
            loss_batch = -np.mean(np.sum(y_one_hot * np.log(proba_sortie), axis=1))
            loss_batches.append(loss_batch)
            
            # Backward propagation
            # Last layer
            delta = proba_sortie - y_one_hot
            grad_W = np.dot(sorties[-1].T, delta) / batch_size
            grad_b = np.mean(delta, axis=0)
            dnn['W'][-1] -= learning_rate * grad_W
            dnn['b'][-1] -= learning_rate * grad_b

            # Hidden layers
            for i in range(2, len(dnn['W'])):
                if i == 2:  # Classification layer
                    RBM = {'W': dnn['W'][-1].T, 'a': dnn['a'][-1], 'b': dnn['b'][-1]}  # Transpose W
                    delta = np.dot(delta, RBM['W']) * sorties[-1] * (1 - sorties[-1])
                else:
                    RBM = {'W': dnn['W'][-i + 1].T, 'a': dnn['a'][-i + 1], 'b': dnn['b'][-i + 1]}
                    delta = np.dot(delta, RBM['W']) * sorties[-i] * (1 - sorties[-i])
                if i == len(dnn['W']) - 1:
                    grad_W = np.dot(X_batch.T, delta)
                else:
                    grad_W = np.dot(sorties[-i - 1].T, delta)
                
                grad_b = np.mean(delta, axis=0)
    
                # Update weights and biases
                dnn['W'][-i - 1] -= learning_rate * grad_W
                dnn['b'][-i - 1] -= learning_rate * grad_b


        # Calculate the cross entropy loss for the epoch
        train_loss = float(np.mean(loss_batches))
        loss.append(train_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}")

        # Check for improvement
        if train_loss < min_loss:
            min_loss = train_loss  # Update minimum loss
            wait = 0  # Reset wait counter
        else:
            wait += 1  # Increment wait counter

        # Check for early stopping
        if wait >= patience:
            print("Early stopping due to no improvement in Loss.")
            break
        
          
    
    if plot:
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
    return dnn


def test_dnn(X, y, dnn, verbose=True):
    _, proba_sortie = entree_sortie_reseau(dnn, X)
    predictions = np.argmax(proba_sortie, axis=1)
    accuracy = np.mean(predictions == y)
    if verbose :
        print(f"Accuracy: {accuracy}")
    return accuracy


def plot_proba(data, dnn, n =10):
    _, pred_labels = entree_sortie_reseau(dnn, data)
    plt.scatter(np.arange(0, n), pred_labels)
    plt.xlabel("Classes")
    plt.ylabel("Predicted probability for each class")
    plt.title("Probabilities by class")
    plt.show(block=False)