from src.data_utils import lire_alpha_digit , lire_mnist 
from src.data_utils import save_model, load_model, plot_error_from_accuracy
from src.rbm import init_RBM, train_RBM, generer_image_RBM
from src.dbn import init_DBN, train_DBN, generate_image_DBN
from src.dnn import init_DNN, pretrain_DNN, retropropagation, test_dnn, plot_proba
import numpy as np

def first_run_experiment():
    """
    Run the first experiment to train a Deep Neural Network (DNN) on the MNIST dataset.
    """
    # Initialize the DNN model
    dnn = init_DNN([p_mnist, 200, 200], output_size=output_dim)
    
    # Pretrain the DNN model
    dnn = pretrain_DNN(X_train, dnn, epochs=epochs_rbm,
                      learning_rate=learning_rate, batch_size=batch_size)
    
    # Train the DNN model using backpropagation
    dnn = retropropagation(X_train, y_train, dnn, epochs=epochs_dnn,
                          learning_rate=learning_rate, batch_size=batch_size, verbose=True, 
                          pretrained=True, save_path='./figs/dnn_mnist.png')
    
    # Test the DNN model
    acc = test_dnn(X_test, y_test, dnn)
    print("DNN accuracy on test dataset:", acc)

    # Plot the probability 
    for k in range(1,4):
        idx = np.where(y_test == k)[0]
        plot_proba(X_test[idx], dnn, k, save_path=f'./figs/proba_{k}.png')

def compare_pretrained_vs_random():
    """
    Compare the performance of a pretrained DNN with a randomly initialized DNN on the MNIST dataset.
    """

    # Initialize the DNN models
    dnn_pretrained = init_DNN([p_mnist, 200, 200], output_size=output_dim)
    dnn_random = init_DNN([p_mnist, 200, 200], output_dim)

    # Pretrain the DNN model
    dnn_pretrained = pretrain_DNN(X_train, dnn_pretrained, epochs=epochs_rbm,
                                  learning_rate=learning_rate, batch_size=batch_size)

    # Train the DNN models using backpropagation
    dnn_pretrained = retropropagation(X_train, y_train, dnn_pretrained,
                                      epochs=epochs_dnn, learning_rate=learning_rate,
                                      batch_size=batch_size,verbose=False, 
                                      pretrained=True, save_path='./figs/pretrained_dnn2.png')
    dnn_random = retropropagation(X_train, y_train, dnn_random,
                                epochs=epochs_dnn, learning_rate=learning_rate,
                                batch_size=batch_size, verbose=False, 
                                pretrained=False, save_path='./figs/random_dnn2.png')

    # Test the DNN models on train dataset
    acc_pretrained = test_dnn(X_train, y_train, dnn_pretrained)
    acc_random = test_dnn(X_train, y_train, dnn_random)
    print("Pretrained DNN accuracy on train dataset:", acc_pretrained)
    print("Randomly initialized DNN accuracy on train dataset:", acc_random)
    # Test the DNN models on test dataset
    acc_pretrained = test_dnn(X_test, y_test, dnn_pretrained)
    acc_random = test_dnn(X_test, y_test, dnn_random)
    print("Pretrained DNN accuracy on test dataset:", acc_pretrained)
    print("Randomly initialized DNN accuracy on test dataset:", acc_random)

    

def analyze_effect_of_layers(layers):
    """
    Analyze the effect of different layer configurations on the performance of pretrained and random DNNs.
    """
    acc_pretrained = []
    acc_random = []
    for layer_count in layers:
        # Initialize the DNN models
        dnn_pretrained = init_DNN([p_mnist] + [200] * layer_count, output_size=output_dim)
        dnn_random = init_DNN([p_mnist] + [200] * layer_count, output_dim)
        
        # Pretrain the DNN models
        dnn_pretrained = pretrain_DNN(X_train, dnn_pretrained, epochs=epochs_rbm,
                                      learning_rate=learning_rate, batch_size=batch_size)
        
        # Train the DNN models using backpropagation
        dnn_pretrained = retropropagation(X_train, y_train, dnn_pretrained,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=True, save_path=f'./figs/pretrained_dnn_layers{layer_count}.png')
        
        dnn_random = retropropagation(X_train, y_train, dnn_random,
                                      epochs=epochs_dnn, learning_rate=learning_rate,
                                      batch_size=batch_size, verbose=False, 
                                      pretrained=False, save_path=f'./figs/random_dnn_layers{layer_count}.png')
        
        # Test the DNN models
        acc_pre = test_dnn(X_test, y_test, dnn_pretrained)
        acc_rand = test_dnn(X_test, y_test, dnn_random)
        acc_pretrained.append(acc_pre)
        acc_random.append(acc_rand)
        
        print(f"Pretrained DNN accuracy with {layer_count} layers:", acc_pre)
        print(f"Randomly initialized DNN accuracy with {layer_count} layers:", acc_rand)
    plot_error_from_accuracy(acc_pretrained, acc_random,
                             layers, 'Layers', save_path='./figs/layers_analysis.png')

def analyze_effect_of_neurons(neurons):
    """
    Analyze the effect of different neuron counts on the performance of pretrained and random DNNs.
    """
    acc_pretrained = []
    acc_random = []
    for neuron_count in neurons:
        # Initialize the DNN models
        dnn_pretrained = init_DNN([p_mnist, neuron_count, neuron_count], output_size=output_dim)
        dnn_random = init_DNN([p_mnist, neuron_count, neuron_count], output_dim)
        
        # Pretrain the DNN models
        dnn_pretrained = pretrain_DNN(X_train, dnn_pretrained, epochs=epochs_rbm,
                                      learning_rate=learning_rate, batch_size=batch_size)
        
        # Train the DNN models using backpropagation
        dnn_pretrained = retropropagation(X_train, y_train, dnn_pretrained,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=True, save_path=f'./figs/pretrained_dnn_neurons{neuron_count}.png')
        
        dnn_random = retropropagation(X_train, y_train, dnn_random,
                                      epochs=epochs_dnn, learning_rate=learning_rate,
                                      batch_size=batch_size, verbose=False, 
                                      pretrained=False, save_path=f'./figs/random_dnn_neurons{neuron_count}.png')
        
        # Test the DNN models
        acc_pre = test_dnn(X_test, y_test, dnn_pretrained)
        acc_rand = test_dnn(X_test, y_test, dnn_random)
        acc_pretrained.append(acc_pre)
        acc_random.append(acc_rand)
        
        print(f"Pretrained DNN accuracy with {neuron_count} neurons per layer:", acc_pre)
        print(f"Randomly initialized DNN accuracy with {neuron_count} neurons per layer:", acc_rand)
    plot_error_from_accuracy(acc_pretrained, acc_random, neurons, 'Neurons', save_path='./figs/neurons_analysis.png')

def analyze_effect_of_training_size(train_sizes):
    """
    Analyze the effect of different training data sizes on the performance of pretrained and random DNNs.
    """
    acc_pretrained = []
    acc_random = []
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    print('hhhhhhhhhhhhhhhh', type(y_train))
    for size in train_sizes:
        # Initialize the DNN models
        dnn_pretrained = init_DNN([p_mnist, 200, 200], output_size=output_dim)
        dnn_random = init_DNN([p_mnist, 200, 200], output_dim)
        
        # Pretrain the DNN models
        dnn_pretrained = pretrain_DNN(X_shuffled[:size], dnn_pretrained, epochs=epochs_rbm,
                                      learning_rate=learning_rate, batch_size=batch_size)
        
        # Train the DNN models using backpropagation
        dnn_pretrained = retropropagation(X_shuffled[:size], y_shuffled[:size], dnn_pretrained,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=True, save_path=f'./figs/pretrained_dnn_datasize{size}.png')
        
        dnn_random = retropropagation(X_shuffled[:size], y_shuffled[:size], dnn_random,
                                      epochs=epochs_dnn, learning_rate=learning_rate,
                                      batch_size=batch_size, verbose=False, 
                                      pretrained=False, save_path=f'./figs/random_dnn_datasize{size}.png')
        
        # Test the DNN models
        acc_pre = test_dnn(X_test, y_test, dnn_pretrained)
        acc_rand = test_dnn(X_test, y_test, dnn_random)
        acc_pretrained.append(acc_pre)
        acc_random.append(acc_rand)
        
        print(f"Pretrained DNN accuracy with training size {size}:", acc_pre)
        print(f"Randomly initialized DNN accuracy with training size {size}:", acc_rand)
    plot_error_from_accuracy(acc_pretrained, acc_random, train_sizes, 'Training Size', save_path='./figs/train_size_analysis.png')

Mnist_image_shame = 28 

x_shape = 20
y_shape = 16

output_dim = 10
epochs_rbm = 100
epochs_dnn = 200
learning_rate = 0.1
batch_size = 128
nb_iterations = 500 
nb_images = 4

# Load the data
data_path = 'data/MNIST/mnist_all.mat'
X_train, y_train = lire_mnist(data_path, np.arange(10), 'train')
X_test, y_test = lire_mnist(data_path, np.arange(10), 'test')
_, p_mnist = X_train.shape








import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="MNIST DNN Experiment")

# Define general arguments for the experiment type and data configuration
parser.add_argument('--experiment_type', choices=['first_run','compare', 'analysis'],
                    help="Type of experiment to run: comparing DNN networks or conducting a detailed analysis.")

# Define arguments for detailed analysis types
parser.add_argument('--analysis_type', choices=['layers', 'neurons', 'train_size'],
                    help="Specify the type of detailed analysis: layers, neurons, or training data size.")

# For the analysis, accept lists of values for layers, neurons, or training sizes
parser.add_argument('--layers', nargs='+', type=int, help="List of layer counts to test, separated by spaces.")
parser.add_argument('--neurons', nargs='+', type=int, help="List of neuron counts per layer to test, separated by spaces.")
parser.add_argument('--train_sizes', nargs='+', type=int, help="List of training sizes to test, separated by spaces.")

args = parser.parse_args()
if args.experiment_type == 'first_run':
    # Here, you would call a function to run the first experiment
    first_run_experiment()
elif args.experiment_type == 'networks':
    # Here, you would call a function to compare a pretrained network with a randomly initialized one.
    compare_pretrained_vs_random()

elif args.experiment_type == 'analysis':
    if args.analysis_type == 'layers':
        # Call your experiment function for varying number of layers, passing the layers list
        analyze_effect_of_layers(args.layers)
    elif args.analysis_type == 'neurons':
        # Call your experiment function for varying neurons per layer, passing the neurons list
        analyze_effect_of_neurons(args.neurons)
    elif args.analysis_type == 'train_size':
        # Call your experiment function for varying training sizes, passing the training sizes list
        analyze_effect_of_training_size(args.train_sizes)




