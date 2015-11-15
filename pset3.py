import numpy as np
import sys
import h5py
import copy
from itertools import groupby
import matplotlib.pyplot as plt

CLASSES = 10
DATASET = 'clean'
ETA_W = 0.001
ETA_B = 0.001
THRESHOLD = 0.0000000001
MAX_ITERATION = 1000
LAMBDA = 0.0000001
#DATASET = 'noisy'

def load_data(filename, dataset):
    try:
        with h5py.File(filename) as data:
            X = data[dataset]['x'][:].astype(np.float64) / 255
            N = X.shape[0]
            X = X.reshape((N, -1))
            if 'y' in data[dataset]:
                y = data[dataset]['y'][:]
                Y = np.zeros((N, CLASSES))
                Y[np.arange(N), y] = 1
                return X, Y
            else:
                return X
    except KeyError:
        print('ERROR: Please place the {} in the same directory as pset3.py'.format(filename))
        sys.exit(1)


def test(X, Y, model):
    return predict(X, model).argmax(-1) == Y.argmax(-1)


def error_rate(X, Y, model):
    return 1 - test(X, Y, model).mean()


def save_submission(filename, yhats):
    assert np.ndim(yhats) == 1
    id_and_prediction = np.vstack([np.arange(len(yhats)).T, yhats]).T
    np.savetxt(filename, id_and_prediction,
               fmt='%d',
               delimiter=',',
               comments='',
               header='Id,Prediction')


def softmax(Z):
    """
    Z is 2D and a softmax is taken over the second dimension for each sample
    separately.
    """
    # Change the dynamic range before normalization, to avoid precision errors
    Z -= Z.max(1)[:, np.newaxis]
    expZ = np.exp(Z)
    return expZ / expZ.sum(1)[:, np.newaxis]


def predict(X, model):
    """
    Evaluate the soft predictions of the model.
    """
    return softmax(np.dot(X, model['weight']) + model['bias'])


def calc_loss(X, Y, model):
    """
    Evaluate the loss (without regularization penalty).
    """
    Z = predict(X, model)
    return -(Y * np.log(Z)).sum() / len(Y)


def gradient_weight(X, Y, model):
    """
    Gradient of the regularized loss with respect to the weights.
    """
    W = model['weight']
    b = model['bias']
    weight_decay = model['weight_decay']

    # YOUR CODE HERE
    # Write the gradient with respect to the weights.
    return np.add(np.subtract(np.dot(np.transpose(predict(X, model)), X), np.dot(np.transpose(Y), X)), 2 * LAMBDA * np.transpose(model['weight'])) #np.zeros((X.shape[1], Y.shape[1]))


def gradient_bias(X, Y, model):
    """
    Gradient of the (regularized) loss with respect to the bias.
    """
    W = model['weight']
    b = model['bias']
    weight_decay = model['weight_decay']

    # YOUR CODE HERE
    # Write the gradient with respect to the bias
    # The following line is just a placeholder
    return np.subtract(np.dot(np.transpose(predict(X, model)), np.ones(len(Y))), np.dot(np.transpose(len(Y)), np.ones(10))) #np.zeros(Y.shape[1])


def get_batch(X, Y, iteration):
    """
    Returns a batch, given which iteration it is.
    """
    offset = 100
    start = iteration * offset % len(Y)
    
    # YOUR CODE HERE
    # This will return the entire data set each iteration. This is costly, so
    # you should experiment with different way of changing this:
    return X[start: start + offset], Y[start: start + offset]

def square_sum(array):
    return np.sum(array ** 2)

def main():
    fn = '{}_mnist.h5'.format(DATASET)

    X, Y = load_data(fn, '/train')
    tX, tY = load_data(fn, '/test')

    print('Dataset:', DATASET)

    SAMPLES, FEATURES = X.shape

    # Weight decay
    weight_decay = 0.0

    # Random number generator for initializing the weights
    rs = np.random.RandomState(2341)
    model = {
        # Initialization of model (you can change this if you'd like)
        'weight': rs.normal(scale=0.01, size=(FEATURES, CLASSES)),
        'bias': np.zeros(CLASSES),
        'weight_decay': weight_decay
    }

    iteration = 0
    old_weight = None
    old_bias = None
    while True:
        # Get the batch (X_i, Y_i)
        bX, bY = get_batch(X, Y, iteration)

        if iteration % 250 == 0:
            loss = calc_loss(bX, bY, model)
            print('{:8} batch loss: {:.3f}'.format(iteration, loss))

        

        # Gradient calculation
        grad_w = gradient_weight(bX, bY, model)
        grad_b = gradient_bias(bX, bY, model)
        # YOUR CODE HERE
        # Write the update rules
        old_weight = copy.deepcopy(model['weight'])
        old_bias = copy.deepcopy(model['bias'])
        model['weight'] += -np.transpose(grad_w * ETA_W)
        model['bias'] += -np.transpose(grad_b * ETA_B)

        # YOUR CODE HERE
        # Change this stopping criterion
        if (square_sum(model['weight'] - old_weight) < THRESHOLD and square_sum(model['bias'] - old_bias) < THRESHOLD) or iteration > MAX_ITERATION:
            break

        iteration += 1

    train_err = error_rate(X, Y, model)
    test_err = error_rate(tX, tY, model)

    print('train error:  {:.2f}%'.format(100 * train_err))
    print('test error:   {:.2f}%'.format(100 * test_err))

    confusion_matrix = [[0 for x in range(10)] for x in range(10)] 

    #print(confusion_matrix)

    # Generate a Kaggle submission file using `model`
    kX = load_data(fn, '/kaggle')
    Yhat = predict(kX, model)
    save_submission('submission_{}.csv'.format(DATASET), Yhat.argmax(-1))

    index = 0
    for index, y_hat in enumerate(predict(X, model).argmax(-1)):
        y = Y[index].tolist().index(1.)
        confusion_matrix[y][y_hat] += 1
    for line in confusion_matrix:
        #print(line)
        print('\t'.join([str(i) for i in line]))

    for i in range(10):
        plt.imshow(model['weight'][:, i].reshape(24, 24))
        plt.show()

    #print(confusion_matrix)


if __name__ == '__main__':
    main()
