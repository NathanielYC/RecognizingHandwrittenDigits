import pickle
import gzip
import os
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    
    # Find the project directory (one level above src)
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Path to the data file
    data_path = os.path.join(project_dir, 'data', 'mnist.pkl.gz')

    # Open the data file
    with gzip.open(data_path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing (training_data, validation_data, test_data).
    
    The training_data is a list of tuples (x, y) representing the training inputs
    and the desired outputs. The validation_data and test_data are lists of
    (x, y) tuples, where x is the input and y is the corresponding digit value."""
    
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position
    and zeroes elsewhere. Used to convert a digit (0...9) into a corresponding
    desired output from the neural network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
