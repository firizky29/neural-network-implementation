from Layer import Layer
from NeuralNetwork import NeuralNetwork
from NeuralNetworkLearning import NeuralNetworkLearning
import numpy as np

from sklearn.neural_network import MLPRegressor

import pytest


def assest_weight_bias(w_actual, b_actual, w_expect, b_expect):
    assert np.square(w_actual-w_expect).sum() < 10e-6
    assert np.square(b_actual-b_expect).sum() < 10e-6


def test_sigmoid():
    l1 = Layer(
        activation="sigmoid",
        is_output=False,
        bias=np.array([0.35, 0.35]),
        weight=np.array([
            [0.15, 0.20],
            [0.25, 0.30]
        ])
    )
    l2 = Layer(
        activation="sigmoid",
        is_output=True,
        bias=np.array([0.6, 0.6]),
        weight=np.array([
            [0.40, 0.45],
            [0.50, 0.55]
        ]),
    )

    nn = NeuralNetwork()
    nn.add_layer(l1)
    nn.add_layer(l2)

    input = np.array([[0.05, 0.1]])
    output = np.array([[0.01, 0.99]])

    learn = NeuralNetworkLearning(nn, learning_rate=0.5)
    learn.run_epoch(input, output)

    # Testing
    expected_weight_l2 = np.array([
        [0.35891648, 0.408666186],
        [0.511301270, 0.561370121]
    ])
    expected_new_b2 = np.array([0.5308, 0.6190])

    expected_weight_l1 = np.array([
        [0.149780716, 0.19956143],
        [0.24975114, 0.29950229]
    ])
    expected_new_b1 = np.array([0.3456, 0.3450])

    assest_weight_bias(l2.get_w(), l2.get_b(),
                       expected_weight_l2, expected_new_b2)
    assest_weight_bias(l1.get_w(), l1.get_b(),
                       expected_weight_l1, expected_new_b1)


def test_linear():
    l1 = Layer(
        activation="linear",
        bias=np.array([.1, .3, .2]),
        weight=np.array([
            [.4, .1],
            [.2, -.8],
            [-.7, .5]
        ]),
        is_output=True
    )

    nn = NeuralNetwork()
    nn.add_layer(l1)

    input = np.array([[3.0, 1.0], [1.0, 2.0]])
    output = np.array(
        [
            [2.0, 0.3, -1.9],
            [1.3, -0.7, 0.1]
        ]
    )

    learn = NeuralNetworkLearning(nn, learning_rate=0.001)
    learn.run_epoch(input, output, batch_size=2, shuffle=False)

    expected_b = np.array([0.1012, 0.3006, 0.1991])
    expected_w = np.array(
        [
            [0.402, 0.101],
            [0.201, -0.799],
            [-0.7019, 0.4987]
        ]
    )

    assest_weight_bias(l1.get_w(), l1.get_b(), expected_w, expected_b)


@pytest.mark.skip(reason="test case belum divalidasi")
def test_relu():
    l = Layer(
        activation="relu",
        bias=np.array([.1, .2, .3]),
        weight=np.array([
            [.4, .7],
            [-.5, .8],
            [.6, -.9]
        ]),
        is_output=True
    )

    nn = NeuralNetwork()
    nn.add_layer(l)

    input = np.array([[-1., .5], [.5, -1.]])
    output = np.array(
        [
            [.1, 1., .1],
            [.1, .1, 1.]
        ]
    )

    learn = NeuralNetworkLearning(nn, learning_rate=0.1)
    learn.run_epoch(input, output, batch_size=2, shuffle=False)

    expected_b = np.array([0.095, 0.21, 0.35])
    expected_w = np.array(
        [
            [0.405, 0.6975],
            [-.51, .805],
            [.625, -.95]
        ]
    )
