from src.data_utils import lire_alpha_digit
import numpy as np
from matplotlib import pyplot as plt

# Load the data
data_path = 'data/binary_alpha_digits/binaryalphadigs.mat'
X = lire_alpha_digit(data_path, np.arange(11,14))
print(X.shape)