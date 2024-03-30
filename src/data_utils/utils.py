import os
import pickle
import matplotlib.pyplot as plt


def save_model(file_name: str, settings: list, model):
    """
    Saves a model to a file within the 'models' directory. The filename incorporates both a base file name
    provided by the user and the network settings.

    :param file_name: Base name for the file.
    :param settings: List of integers representing the number of neurons in each layer.
    :param model: Trained model to save.
    """
    # Ensure the 'models' directory exists
    os.makedirs("./models", exist_ok=True)
    
    # Convert settings to a string and append to file_name
    settings_str = "".join(str(s) for s in settings)
    filename_with_settings = f"{file_name}{settings_str}.pkl"
    
    filepath = os.path.join("./models", filename_with_settings)
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

def load_model(file_name: str, settings: list):
    """
    Loads a model from a file within the 'models' directory. The filename incorporates both a base file name
    provided by the user and the network settings.

    :param file_name: Base name for the file.
    :param settings: List of integers representing the number of neurons in each layer.
    :return: Loaded model or None if the model cannot be loaded.
    """
    # Convert settings to a string and append to file_name
    settings_str = "".join(str(s) for s in settings)
    filename_with_settings = f"{file_name}{settings_str}.pkl"
    
    filepath = os.path.join("./models", filename_with_settings)
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
            print(f"Model loaded from {filepath}")
            return model
    except FileNotFoundError:
        print(f"Warning: The file {filepath} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


def plot_error_from_accuracy(accuracy1, accuracy2, parameters, parameter_name, save_path=None):
    # Calculate error rates from accuracies
    error_rate1 = [1 - a for a in accuracy1]
    error_rate2 = [1 - a for a in accuracy2]

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    # Plot for the pretrained network
    ax[0].plot(parameters, error_rate1, color='blue', linestyle='-', marker='o', markersize=8, label='Pretrained DNN')
    ax[0].set_title("Impact of " + parameter_name + " on pretrained DNN", fontsize=14)
    ax[0].set_xlabel(parameter_name, fontsize=12)
    ax[0].set_ylabel("Error Rate", fontsize=12)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend()

    # Plot for the neural network without pretraining
    ax[1].plot(parameters, error_rate2, color='red', linestyle='-', marker='x', markersize=8, label='DNN')
    ax[1].set_title("Impact of " + parameter_name + " on random DNN", fontsize=14)
    ax[1].set_xlabel(parameter_name, fontsize=12)
    ax[1].set_ylabel("Error Rate", fontsize=12)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend()

    plt.suptitle(f"Comparing error rate of both models by {parameter_name}", fontsize=16)
    plt.tight_layout(pad=5.0)

     # If a save path is provided, save the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show(block=False)
