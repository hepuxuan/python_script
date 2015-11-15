import numpy as np
import h5py
import copy
import pylab
from numpy.linalg import inv
from sklearn import cross_validation

SEISMIC_COLUMN_INDEX = 0
TEMPERATURE_COLUMN_INDEX = 1


def degexpand(X, deg, C=None):
    """
    Prepares data matrix with a column of ones and polynomials of specified
    degree.

    Parameters
    ----------
    X : 2D array
        n x d data matrix (row per example)
    deg : integer
        Degree of polynomial
    C : 1D array
        Scaling weights. If not specifed, weights are calculated to fit each
        columns of X in [-1, 1].
        Note: It is shown in problem set 1 that this normalization does
        not affect linear regression, as long as it is applied
        consistently to training *and* test data.

    Returns
    -------
    out_X : 2D array
        n x (2 * d + 1) data matrix (row per example)
        The output is arranged as follows:
            - X[:, 0] is all ones
            - X[:, 1] is x_1
            - X[:, 2] is x_1^2
            - ...
            - X[:, deg] is x_1^deg
            - X[:, deg+1] is x_2
            - X[:, deg+2] is x_2^2
            - etc.
    C : 1D array
        Scaling weights that were used. Useful if no C was specified.
    """
    assert X.ndim == 2
    #m, n = X.shape
    n, m = X.shape

    # Make polynomials
    out_X = (X[..., np.newaxis] ** (1. + np.arange(deg))).reshape(n, -1)

    # Add column of ones
    out_X = np.concatenate([np.ones((out_X.shape[0], 1)), out_X], axis=1)

    if C is None:
        C = abs(out_X).max(axis=0)
    else:
        assert np.shape(C) == (out_X.shape[1],), "C must match outgoing matrix"

    out_X /= C
    return out_X, C


def test_ws(X, y, wrange, lossfunc):
    """
    Tests a range of parameters against a loss function.

    Parameters
    ----------
    X : 2D array
        N x d+1 design matrix (row per example)
    y : 1D array
        Observed function values
    wrange : list of arrays
        A list of arrays with parameters that should be tested. The value
        wrange[j] is a range over which the j-th coefficient has to run.
    lossfunc : function
        Function pointer to compute empirical loss. lossfunc is assumed to
        take two arguments: yhat (estimated) and y (observed).

    Returns
    -------
    losses : 2D array
        losses[i, j] is the value of the loss for the corresponding set of
        parameter (regression coefficient) values.
    """
    assert np.ndim(X) == 2, "X must be 2D"
    assert X.shape[1] == len(wrange), "Sizes of X and wrange should match"

    ws = np.reshape(np.meshgrid(*wrange, indexing='ij'), (len(wrange), -1))
    yhat = np.dot(X, ws)

    loss = np.zeros(yhat.shape[1])
    for i in range(len(loss)):
        loss[i] = lossfunc(yhat[:, i], y)

    return loss.reshape([len(w) for w in wrange])

def get_beta_search(train_x, train_y, dim):
    
    beta_matrix = [ np.arange(-10, 10, 0.5) for _ in range(dim) ]

    losses = test_ws(train_x, train_y, beta_matrix, abs_loss)

    loc = np.where(losses == np.amin(losses))

    beta = []
    index = 0
    for i in loc:
        beta.append(beta_matrix[index][i])
        index += 1
    return beta

def get_beta_search_linear(train_x, train_y):

    return get_beta_search(train_x, train_y, 2)

def get_beta_search_quad(train_x, train_y):

    return get_beta_search(train_x, train_y, 3)

def get_data_size(data):
    return data.shape[0]

def get_beta(train_x, train_y):
    train_xt = np.transpose(train_x)
    return np.dot(np.dot(inv(np.dot(train_xt, train_x)), train_xt), train_y)

def degexpand_func_linear(input):
    return degexpand(input, 1)[0]

def degexpand_func_quad(input):
    return degexpand(input, 2)[0]

def get_bias_square_sum(expected_y, y):
    return np.sum(np.square(np.subtract(expected_y, y)))

def get_bias_abs_sum(expected_y, y):
    return np.sum(np.abs(np.subtract(expected_y, y)))

def get_expected_temperature(train_seismic, test_seismic, train_temperature, degexpand_func, get_beta_func):
    train_x = degexpand_func(train_seismic)
    beta = get_beta_func(train_x, train_temperature)
    test_x = degexpand_func(test_seismic)
    return np.dot(test_x, beta)

def get_degexpand_func_bias_mean(dataset, degexpand_func, get_beta_func = get_beta, bias_func = get_bias_square_sum):
    train_seismic = dataset['train_set']['seismic']
    train_temperature = dataset['train_set']['temperature']

    test_seismic = dataset['test_set']['seismic']
    test_temperature = dataset['test_set']['temperature']

    expected_temperature = get_expected_temperature(train_seismic, test_seismic, train_temperature, degexpand_func, get_beta_func)

    return bias_func(expected_temperature, test_temperature)/get_data_size(dataset['test_set']['seismic'])

def get_k_fold_bias_mean(input, degexpand_func, get_beta_func = get_beta, bias_func = get_bias_square_sum):
    return np.mean([ get_degexpand_func_bias_mean(dataset, degexpand_func, get_beta_func, bias_func) for dataset in input ])

def process_raw_data(raw_data, train_index = None, test_index = None):
    if train_index is None:
        train_set = raw_data[0:-1]
    else:
        train_set = raw_data[train_index]

    if test_index is None:
        test_set = raw_data[0:-1]
    else:
        test_set = raw_data[test_index]

    get_data_map = lambda input: { 'seismic': input[:, [SEISMIC_COLUMN_INDEX]], 'temperature': input[:, [TEMPERATURE_COLUMN_INDEX]]}
    return {"train_set": get_data_map(train_set), "test_set": get_data_map(test_set)}

def get_cross_validation_dataset(dateset):
    kf = cross_validation.KFold(n = get_data_size(dateset), n_folds = 10, shuffle = True)
    return [ process_raw_data(dateset, train_index, test_index) for train_index, test_index in kf ]

def as_sorted_array(matrix):
    return np.sort(np.squeeze(matrix))

def abs_loss(yhat, y):
    return np.sum(np.abs(np.subtract(yhat, y)))

def main():

    f = None
    try:
        f = h5py.File('/Users/xiaoleilin/Desktop/PhD/HMK/TTIC_ML/Assignment/1/data/fortgauss.h5', 'r')

        raw_data08 = f['data08'][()]
        raw_data09 = f['data09'][()]
        
    except OSError:
        print('missing h5 file.')
        return
    finally:
        if f is not None:
            f.close()
    dataset_08_list = get_cross_validation_dataset(raw_data08)
    dataset_08_list = get_cross_validation_dataset(raw_data08)
    dataset_08 = process_raw_data(raw_data09)
    dataset_09 = process_raw_data(raw_data09)

    print(get_k_fold_bias_mean(dataset_08_list, degexpand_func_linear)) # 1.06881467246

    print(get_k_fold_bias_mean(dataset_08_list, degexpand_func_quad)) #0.959759724444

    print(get_degexpand_func_bias_mean(dataset_09, degexpand_func_quad)) #1.11285607781

    print(get_k_fold_bias_mean(dataset_08_list, degexpand_func_linear, get_beta_search_linear, get_bias_abs_sum)) #1.16520737634

    print(get_k_fold_bias_mean(dataset_08_list, degexpand_func_quad, get_beta_search_quad, get_bias_abs_sum)) #1.2109710294

    print(get_degexpand_func_bias_mean(dataset_08, degexpand_func_linear, get_beta_search_linear)) #2.32285301004

    print(get_degexpand_func_bias_mean(dataset_09, degexpand_func_linear, get_beta_search_linear))  #1.95489144174

    #plot:
    #x_09 = degexpand_func_quad(dataset_09['test_set']['seismic'])
    #beta_09 = get_beta(x_09, dataset_09['test_set']['temperature'])
    #except_temperature_09 = np.dot(x_09, beta_09)

    pylab.xlabel("X")
    pylab.ylabel("Y")

    #pylab.plot(as_sorted_array(dataset_09['test_set']['seismic']), as_sorted_array(except_temperature_09))
    #pylab.plot(dataset_09['test_set']['seismic'], dataset_09['test_set']['temperature'], 'ro')

    x_08 = degexpand_func_linear(dataset_08['test_set']['seismic'])
    beta_08 = get_beta_search_linear(x_08, dataset_08['test_set']['temperature'])
    except_temperature_08 = np.dot(x_08, beta_08)

    pylab.plot(as_sorted_array(dataset_08['test_set']['seismic']), as_sorted_array(except_temperature_08))
    pylab.plot(dataset_08['test_set']['seismic'], dataset_08['test_set']['temperature'], 'ro')
    pylab.show()

main()
