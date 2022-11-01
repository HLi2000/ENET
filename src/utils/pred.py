import itertools

import torch


def ordinariser(inputs, n_classes=4):
    """ Label ordinarizer

    Converts multiclass label (0,1,2,...) for k-classes into ordinal representation

    # Arguments
        inputs: input labels.
            - e.g. [0,2,1,3,...]
        k_lim: integer, the number of classes.
    # Returns
        output labels in ordinal form.
            - e.g.[[0,0,0],[1,1,0],[1,0,0],[1,1,1],...]

    """
    outputs = torch.tensor([ [ 0 for i in range(n_classes-1) ] for j in range(len(inputs)) ]).to(inputs.device)

    for i, val in enumerate(inputs):
        if val > n_classes or val < 0:
            raise ValueError("Value out of range!")
        for j in range(int(val)):
            outputs[i][j] = 1

    return outputs

def ordinariser_reversed(inputs):
    """ Reverse from ordinary labelling to multiclass labels

    Revese the ordinal labels back to multiclass labels

    # Arguments
        inputs: ordinal labels.
            - e.g.[[0,0,0],[1,1,0],...]
    # Returns
        output labels in multiclass label
            - e.g.[0,2,...]

    """
    outputs = []

    for i, val in enumerate(inputs):
        out = torch.sum(val, dtype=torch.int32)
        # if out > k_lim or out < 0:
        #     raise Exception("[ERROR] Value out of range!")
        outputs.append(out)

    return torch.stack(outputs).to(inputs.device)

def proba_ordinal_to_categorical(inputs):
    """ Orinal -> Categorical Probability convertion

    Convert ordinal probas (k-1) to categorical (k-classes) probas

    # Arguments
        inputs: ordinal probabilities (k-1 classes)
            - e.g.[0.123,0.2321,0.888]
    # Returns
        output categorical probabilities (k classes)
            - e.g.[0.877, 0.0944517, 0.0031974095999999994, 0.0253508904]

    """
    k = len(inputs[0]) + 1
    outputs = []
    for input in inputs:
        output = [0 for i in range(k)]
        for c in range(0, k):
            if c is 0:
                output[0] = 1.0 - input[0]
                p_cond = input[0]
            elif c is (k-1):
                output[k-1] = p_cond
            else:
                output[c] = (1.0 - input[c]) * p_cond
                p_cond = input[c] * p_cond
        outputs.append(output)
    return torch.tensor(outputs).to(inputs.device)

def convolve_many(arrays):
    """
    Convolve a list of 1d float arrays together, using FFTs.
    The arrays need not have the same length, but each array should
    have length at least 1.

    """
    result_length = 1 + sum((len(array) - 1) for array in arrays)

    # Copy each array into a 2d array of the appropriate shape.
    rows = torch.zeros((len(arrays), result_length))
    for i, array in enumerate(arrays):
        rows[i, :len(array)] = array

    # Transform, take the product, and do the inverse transform
    # to get the convolution.
    fft_of_rows = torch.fft.fft(rows)
    fft_of_convolution = fft_of_rows.prod(axis=0)
    convolution = torch.fft.ifft(fft_of_convolution)

    # Assuming real inputs, the imaginary part of the output can
    # be ignored.
    return convolution.real