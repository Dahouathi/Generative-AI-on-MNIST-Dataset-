from src.data_utils import lire_alpha_digit , lire_mnist 
from src.data_utils import save_model, load_model, plot_error_from_accuracy
from src.rbm import init_RBM, train_RBM, generer_image_RBM
from src.dbn import init_DBN, train_DBN, generate_image_DBN
from src.dnn import init_DNN, pretrain_DNN, retropropagation, test_dnn, box_plot_proba
import numpy as np




def rbm_alphadigits_experiment(input_list):
    """
    Run an experiment to train a Restricted Boltzmann Machine (RBM) on the alphadigits dataset.
    """
    for i in input_list:
        X = lire_alpha_digit(alphadigit_path, np.array([char_to_num[i]]))
        _, p = X.shape
        # Initialize the RBM model
        rbm = init_RBM(p, 200)
        
        # Train the RBM model
        train_RBM(X, rbm, epochs=epochs_rbm, learning_rate=learning_rate, batch_size=batch_size_rbm)
        
        # Generate images using the RBM model
        generer_image_RBM(rbm, nb_images=nb_images, x_shape=x_shape, y_shape=y_shape,
                           nb_iterations=nb_iterations, Plot=True, save_path='./figs/rbm_alphadigits{}.png'.format(i))

def rbm_mixed_experiment(input_list):
    """
    Run an experiment to train a Restricted Boltzmann Machine (RBM) on a mixed dataset of MNIST and alphadigits.
    """
    input_list_ = [char_to_num[i] for i in input_list]
    X = lire_alpha_digit(alphadigit_path, np.array(input_list_))
    _, p = X.shape
    # Initialize the RBM model
    rbm = init_RBM(p, 200)
    
    # Train the RBM model
    train_RBM(X, rbm, epochs=epochs_rbm, learning_rate=learning_rate, batch_size=batch_size_rbm)

    input_list_str = '_'.join(str(i) for i in input_list)
    # Generate images using the RBM model
    generer_image_RBM(rbm, nb_images=nb_images, x_shape=x_shape, y_shape=y_shape,
                      nb_iterations=nb_iterations, Plot=True, save_path='./figs/rbm_mixed{}.png'.format(input_list_str))
def dbn_experiment(input_list, layers, neurones):
    """
    Run an experiment to train a Deep Belief Network (DBN) on the MNIST dataset.
    """
    for i in input_list:
        X = lire_alpha_digit(alphadigit_path, np.array([char_to_num[i]]))
        _, p = X.shape
        # Initialize the DBN model with the specified layers and neurones
        dbn = init_DBN([p] + [neurones[0] for _ in range(layers[0])])
        
        # Train the DBN model
        dbn = train_DBN(X, dbn, epochs=epochs_rbm, learning_rate=learning_rate, batch_size=batch_size_rbm)
        
        # Generate images using the DBN model
        generate_image_DBN(dbn, nb_images=nb_images, x_shape=x_shape, y_shape=y_shape,
                           nb_iterations=nb_iterations, Plot=True, save_path='./figs/dbn_digits{}{}{}.png'.format(layers, neurones, i))

def dbn_mixed_experiment(input_list, layers, neurones):
    """
    Run an experiment to train a Deep Belief Network (DBN) on a mixed dataset of MNIST and alphadigits.
    """
    input_list_ = [char_to_num[i] for i in input_list]
    X = lire_alpha_digit(alphadigit_path, np.array(input_list_))
    _, p = X.shape
    # Initialize the DBN model with the specified layers and neurones
    dbn = init_DBN([p] + [neurones[0] for _ in range(layers[0])])
    
    # Train the DBN model
    dbn = train_DBN(X, dbn, epochs=epochs_rbm, learning_rate=learning_rate, batch_size=batch_size_rbm)

    input_list_str = '_'.join(str(i) for i in input_list)
    # Generate images using the DBN model
    generate_image_DBN(dbn, nb_images=nb_images, x_shape=x_shape, y_shape=y_shape,
                      nb_iterations=nb_iterations, Plot=True, save_path='./figs/dbn_mixed{}{}{}.png'.format(layers,neurones,input_list_str))

def first_run_experiment():
    """
    Run the first experiment to train a Deep Neural Network (DNN) on the MNIST dataset.
    """
    # Define model settings
    settings = [p_mnist, 200, 200]  # Example settings, adjust as needed
    
    # Check for a fully trained model first
    fully_trained_model_name = "dnn_mnist_fully_trained"
    dnn = load_model(fully_trained_model_name, settings)
    if dnn is not None:
        print("Fully trained model loaded.")
    else:
        # If not found, check for a pretrained model
        pretrained_model_name = "dnn_mnist_pretrained"
        dnn = load_model(pretrained_model_name, settings)
        if dnn is None:
            print("Pretrained model not found. Initializing and pretraining a new model.")
            # Initialize the DNN model
            dnn = init_DNN(settings, output_size=output_dim)
            
            # Pretrain the DNN model
            dnn = pretrain_DNN(X_train, dnn, epochs=epochs_rbm,
                              learning_rate=learning_rate, batch_size=batch_size)
            
            # Save the pretrained model
            save_model(pretrained_model_name, settings, dnn)
            print("Pretrained model saved.")
        else:
            print("Pretrained model loaded.")
        
        # Train the DNN model using backpropagation
        dnn = retropropagation(X_train, y_train, dnn, epochs=epochs_dnn,
                              learning_rate=learning_rate, batch_size=batch_size, verbose=True, 
                              pretrained=True, save_path='./figs/dnn_mnist.png')
        
        # Save the fully trained model
        save_model(fully_trained_model_name, settings, dnn)
        print("Fully trained model saved.")
    
    # Test the DNN model
    acc = test_dnn(X_test, y_test, dnn)
    print("DNN accuracy on test dataset:", acc)

    # Plot the probability for classes 1, 2, and 3
    for k in range(0, 9):
        idx = np.where(y_test == k)[0]
        box_plot_proba(X_test[idx], dnn, k, save_path=f'./figs/proba_{k}.png')

def compare_pretrained_vs_random():
    """
    Compare the performance of a pretrained DNN with a randomly initialized DNN on the MNIST dataset.
    """
    settings = [p_mnist, 200, 200]  # Model settings, adjust as needed

    # Attempt to load an existing fully trained pretrained model
    pretrained_model_name = "dnn_mnist_pretrained_fully_trained"
    dnn_pretrained = load_model(pretrained_model_name, settings)
    if dnn_pretrained is None:
        # If not found, check for a pretrained model
        pretrained_model_name = "dnn_mnist_pretrained"
        dnn_pretrained = load_model(pretrained_model_name, settings)
        if dnn_pretrained is None:
            print("Pretrained model not found. Initializing and pretraining a new model.")
            dnn_pretrained = init_DNN(settings, output_size=output_dim)
            dnn_pretrained = pretrain_DNN(X_train, dnn_pretrained, epochs=epochs_rbm,
                                          learning_rate=learning_rate, batch_size=batch_size)
            save_model(pretrained_model_name, settings, dnn_pretrained)
            print("Pretrained model saved.")
        else:
            print("Pretrained model loaded.")
        
        # Train the DNN model using backpropagation
        dnn_pretrained = retropropagation(X_train, y_train, dnn_pretrained,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=True, save_path='./figs/pretrained_dnn2.png')
        
        # Save the fully trained model
        save_model(pretrained_model_name + "_fully_trained", settings, dnn_pretrained)
        print("Fully trained pretrained model saved.")
    else:
        print("Fully trained pretrained model loaded.")

    # Check for a fully trained (randomly initialized) model
    random_model_name = "dnn_mnist_random_fully_trained"
    dnn_random = load_model(random_model_name, settings)
    if dnn_random is None:
        print("Randomly initialized model not found. Initializing a new model.")
        dnn_random = init_DNN(settings, output_size=output_dim)
        
        # Train the DNN model using backpropagation
        dnn_random = retropropagation(X_train, y_train, dnn_random,
                                      epochs=epochs_dnn, learning_rate=learning_rate,
                                      batch_size=batch_size, verbose=False, 
                                      pretrained=False, save_path='./figs/random_dnn2.png')
        
        # Save the fully trained model
        save_model(random_model_name, settings, dnn_random)
        print("Fully trained randomly initialized model saved.")
    else:
        print("Fully trained randomly initialized model loaded.")

    # Test the DNN models
    acc_pretrained_train = test_dnn(X_train, y_train, dnn_pretrained)
    acc_random_train = test_dnn(X_train, y_train, dnn_random)
    print("Pretrained DNN accuracy on train dataset:", acc_pretrained_train)
    print("Randomly initialized DNN accuracy on train dataset:", acc_random_train)

    acc_pretrained_test = test_dnn(X_test, y_test, dnn_pretrained)
    acc_random_test = test_dnn(X_test, y_test, dnn_random)
    print("Pretrained DNN accuracy on test dataset:", acc_pretrained_test)
    print("Randomly initialized DNN accuracy on test dataset:", acc_random_test)


    

def analyze_effect_of_layers(layers):
    """
    Analyze the effect of different layer configurations on the performance of pretrained and random DNNs.
    """
    acc_pretrained = []
    acc_random = []
    for layer_count in layers:
        settings = [p_mnist] + [200] * layer_count
        pretrained_model_name = f"dnn_mnist_pretrained_layers{layer_count}"
        random_model_name = f"dnn_mnist_random_layers{layer_count}"

        # Check for existing fully trained pretrained model
        dnn_pretrained = load_model(pretrained_model_name + "_fully_trained", settings)
        if dnn_pretrained is None:
            # Check for existing pretrained model
            dnn_pretrained = load_model(pretrained_model_name, settings)
            if dnn_pretrained is None:
                print(f"Pretrained model with {layer_count} layers not found. Initializing and pretraining a new model.")
                dnn_pretrained = init_DNN(settings, output_size=output_dim)
                dnn_pretrained = pretrain_DNN(X_train, dnn_pretrained, epochs=epochs_rbm,
                                              learning_rate=learning_rate, batch_size=batch_size)
                save_model(pretrained_model_name, settings, dnn_pretrained)
                print(f"Pretrained model with {layer_count} layers saved.")
            else:
                print(f"Pretrained model with {layer_count} layers loaded.")

            # Train the DNN model using backpropagation
            dnn_pretrained = retropropagation(X_train, y_train, dnn_pretrained,
                                              epochs=epochs_dnn, learning_rate=learning_rate,
                                              batch_size=batch_size, verbose=False, 
                                              pretrained=True, save_path=f'./figs/pretrained_dnn_layers{layer_count}.png')
            
            # Save the fully trained model
            save_model(pretrained_model_name + "_fully_trained", settings, dnn_pretrained)
            print(f"Fully trained pretrained model with {layer_count} layers saved.")
        else:
            print(f"Fully trained pretrained model with {layer_count} layers loaded.")

        # Check for existing fully trained randomly initialized model
        dnn_random = load_model(random_model_name + "_fully_trained", settings)
        if dnn_random is None:
            # Check for existing randomly initialized model
            dnn_random = load_model(random_model_name, settings)
            if dnn_random is None:
                print(f"Randomly initialized model with {layer_count} layers not found. Initializing a new model.")
                dnn_random = init_DNN(settings, output_size=output_dim)
                print(f"Randomly initialized model with {layer_count} layers saved.")
            else:
                print(f"Randomly initialized model with {layer_count} layers loaded.")

            # Train the DNN model using backpropagation
            dnn_random = retropropagation(X_train, y_train, dnn_random,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=False, save_path=f'./figs/random_dnn_layers{layer_count}.png')

            # Save the fully trained model
            save_model(random_model_name + "_fully_trained", settings, dnn_random)
            print(f"Fully trained randomly initialized model with {layer_count} layers saved.")
        else:
            print(f"Fully trained randomly initialized model with {layer_count} layers loaded.")

        # Test the DNN models
        acc_pre = test_dnn(X_test, y_test, dnn_pretrained)
        acc_rand = test_dnn(X_test, y_test, dnn_random)
        acc_pretrained.append(acc_pre)
        acc_random.append(acc_rand)

        print(f"Pretrained DNN accuracy with {layer_count} layers:", acc_pre)
        print(f"Randomly initialized DNN accuracy with {layer_count} layers:", acc_rand)

    plot_error_from_accuracy(acc_pretrained, acc_random, layers, 'Layers', save_path='./figs/layers_analysis.png')

def analyze_effect_of_neurons(neurons):
    """
    Analyze the effect of different neuron counts on the performance of pretrained and random DNNs.
    """
    acc_pretrained = []
    acc_random = []
    for neuron_count in neurons:
        settings = [p_mnist, neuron_count, neuron_count]
        pretrained_model_name = f"dnn_mnist_pretrained_neurons{neuron_count}"
        random_model_name = f"dnn_mnist_random_neurons{neuron_count}"

        # Check for existing fully trained pretrained model
        dnn_pretrained = load_model(pretrained_model_name + "_fully_trained", settings)
        if dnn_pretrained is None:
            # Check for existing pretrained model
            dnn_pretrained = load_model(pretrained_model_name, settings)
            if dnn_pretrained is None:
                print(f"Pretrained model with {neuron_count} neurons not found. Initializing and pretraining a new model.")
                dnn_pretrained = init_DNN(settings, output_size=output_dim)
                dnn_pretrained = pretrain_DNN(X_train, dnn_pretrained, epochs=epochs_rbm,
                                              learning_rate=learning_rate, batch_size=batch_size)
                save_model(pretrained_model_name, settings, dnn_pretrained)
            else:
                print(f"Pretrained model with {neuron_count} neurons loaded.")

            # Train the DNN models using backpropagation
            dnn_pretrained = retropropagation(X_train, y_train, dnn_pretrained,
                                              epochs=epochs_dnn, learning_rate=learning_rate,
                                              batch_size=batch_size, verbose=False, 
                                              pretrained=True, save_path=f'./figs/pretrained_dnn_neurons{neuron_count}.png')

            # Save the fully trained models after backpropagation
            save_model(pretrained_model_name + "_fully_trained", settings, dnn_pretrained)
        else:
            print(f"Fully trained pretrained model with {neuron_count} neurons loaded.")

        # Check for existing fully trained randomly initialized model
        dnn_random = load_model(random_model_name + "_fully_trained", settings)
        if dnn_random is None:
            # Check for existing randomly initialized model
            dnn_random = load_model(random_model_name, settings)
            if dnn_random is None:
                print(f"Randomly initialized model with {neuron_count} neurons not found. Initializing a new model.")
                dnn_random = init_DNN(settings, output_size=output_dim)
            else:
                print(f"Randomly initialized model with {neuron_count} neurons loaded.")

            # Train the DNN models using backpropagation
            dnn_random = retropropagation(X_train, y_train, dnn_random,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=False, save_path=f'./figs/random_dnn_neurons{neuron_count}.png')

            # Save the fully trained models after backpropagation
            save_model(random_model_name + "_fully_trained", settings, dnn_random)
        else:
            print(f"Fully trained randomly initialized model with {neuron_count} neurons loaded.")

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
    
    for size in train_sizes:
        settings = [p_mnist, 200, 200, size]  # Including training size in settings for unique model identification
        pretrained_model_name = f"dnn_mnist_pretrained_datasize{size}"
        random_model_name = f"dnn_mnist_random_datasize{size}"

        # Check for existing fully trained pretrained model
        dnn_pretrained = load_model(pretrained_model_name + "_fully_trained", settings)
        if dnn_pretrained is None:
            # Check for existing pretrained model
            dnn_pretrained = load_model(pretrained_model_name, settings)
            if dnn_pretrained is None:
                print(f"Pretrained model with training size {size} not found. Initializing and pretraining a new model.")
                dnn_pretrained = init_DNN([p_mnist, 200, 200], output_size=output_dim)
                dnn_pretrained = pretrain_DNN(X_shuffled[:size], dnn_pretrained, epochs=epochs_rbm,
                                              learning_rate=learning_rate, batch_size=batch_size)
                save_model(pretrained_model_name, settings, dnn_pretrained)
            else:
                print(f"Pretrained model with training size {size} loaded.")

            # Train the DNN models using backpropagation
            dnn_pretrained = retropropagation(X_shuffled[:size], y_shuffled[:size], dnn_pretrained,
                                              epochs=epochs_dnn, learning_rate=learning_rate,
                                              batch_size=batch_size, verbose=False, 
                                              pretrained=True, save_path=f'./figs/pretrained_dnn_datasize{size}.png')

            # Save the fully trained models after backpropagation
            save_model(pretrained_model_name + "_fully_trained", settings, dnn_pretrained)
        else:
            print(f"Fully trained pretrained model with training size {size} loaded.")

        # Check for existing fully trained randomly initialized model
        dnn_random = load_model(random_model_name + "_fully_trained", settings)
        if dnn_random is None:
            # Check for existing randomly initialized model
            dnn_random = load_model(random_model_name, settings)
            if dnn_random is None:
                print(f"Randomly initialized model with training size {size} not found. Initializing a new model.")
                dnn_random = init_DNN([p_mnist, 200, 200], output_size=output_dim)
            else:
                print(f"Randomly initialized model with training size {size} loaded.")

            # Train the DNN models using backpropagation
            dnn_random = retropropagation(X_shuffled[:size], y_shuffled[:size], dnn_random,
                                          epochs=epochs_dnn, learning_rate=learning_rate,
                                          batch_size=batch_size, verbose=False, 
                                          pretrained=False, save_path=f'./figs/random_dnn_datasize{size}.png')

            # Save the fully trained models after backpropagation
            save_model(random_model_name + "_fully_trained", settings, dnn_random)
        else:
            print(f"Fully trained randomly initialized model with training size {size} loaded.")

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
batch_size_rbm = 10
nb_iterations = 500 
nb_images = 4

# Load the data
MNIST_path = 'data/MNIST/mnist_all.mat'
X_train, y_train = lire_mnist(MNIST_path, np.arange(10), 'train')
X_test, y_test = lire_mnist(MNIST_path, np.arange(10), 'test')
_, p_mnist = X_train.shape

alphadigit_path = 'data/binary_alpha_digits/binaryalphadigs.mat'

# Define the mapping between characters and numbers
char_to_num = {str(i): i for i in range(10)}  # Digits '0'-'9'
char_to_num.update({chr(65+i-10): (i) for i in range(10, 36)})  # Letters 'A'-'Z'
# Reverse mapping
num_to_char = {i: str(i) for i in range(10)}  # Digits 0-9
num_to_char.update({i: chr(65+i-10) for i in range(10, 36)})  # Letters A-Z







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

# New arguments for RBM and DBN
parser.add_argument('--rbm', choices=['digits', 'alphabets', 'mix_digits', 'mix_alphabets', 'mix_both'],
                    help="Run RBM experiments. Choose one: 'digits', 'alphabets', 'mix_digits', 'mix_alphabets', 'mix_both'.")
parser.add_argument('--rbm_input', nargs='+', type=str, help="List of numbers specifying input for RBM experiments, separated by spaces.")

# New arguments for RBM and DBN
parser.add_argument('--dbn', choices=['digits', 'alphabets', 'mix'],
                    help="Run DBN experiments. Choose one: 'digits', 'alphabets', 'mix_digits', 'mix_alphabets', 'mix_both'.")
parser.add_argument('--dbn_input', nargs='+', type=str, help="List of numbers specifying input for RBM experiments, separated by spaces.")

parser.add_argument('--layer', nargs='+', type=int, default=[2], help="List of numbers specifying layers for DBN experiments, separated by spaces.")
parser.add_argument('--neurone', nargs='+', type=int, default=[200], help="List of numbers specifying neurones for each layer in DBN experiments, separated by spaces.")


args = parser.parse_args()

if args.rbm:
    if args.rbm_input:
        
        if args.rbm == 'digits':
            # Run the RBM experiment for the MNIST dataset
            rbm_alphadigits_experiment(args.rbm_input)
        elif args.rbm == 'alphabets':
            # Run the RBM experiment for the alphadigits dataset
            rbm_alphadigits_experiment(args.rbm_input)
        elif args.rbm == 'mix_digits' or args.rbm == 'mix_alphabets' or args.rbm == 'mix_both':
            # Run the RBM experiment for the mixed dataset of MNIST and alphadigits
            rbm_mixed_experiment(args.rbm_input)
elif args.dbn:
    if args.dbn_input:
        if args.dbn == 'digits' or args.dbn == 'alphabets':
            # Run the DBN experiment for the MNIST dataset
            dbn_experiment(args.dbn_input, args.layer, args.neurone)
        elif args.dbn == 'mix':
            # Run the DBN experiment for the mixed dataset of MNIST and alphadigits
            dbn_mixed_experiment(args.dbn_input, args.layer, args.neurone)
elif args.experiment_type == 'first_run':
    # Here, you would call a function to run the first experiment
    first_run_experiment()
elif args.experiment_type == 'compare':
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
        print("Script started")
        # Call your experiment function for varying training sizes, passing the training sizes list
        analyze_effect_of_training_size(args.train_sizes)




