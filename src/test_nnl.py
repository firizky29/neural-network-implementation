from Layer import Layer
from NeuralNetwork import NeuralNetwork
from NeuralNetworkLearning import NeuralNetworkLearning
import numpy as np

import pytest


def assest_weight_bias(w_actual, b_actual, w_expect, b_expect):
    assert np.square(w_actual-w_expect).sum() < 1e-8
    assert np.square(b_actual-b_expect).sum() < 1e-8


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
    first_err = learn.calculate_error(input, output)

    learn.run_epoch(input, output)
    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

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
    first_err = learn.calculate_error(input, output)

    learn.run_epoch(input, output, batch_size=2, shuffle=False)
    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

    expected_b = np.array([0.1008, 0.3006, 0.1991])
    expected_w = np.array(
        [
            [0.402, 0.101],
            [0.201, -0.799],
            [-0.7019, 0.4987]
        ]
    )

    assest_weight_bias(l1.get_w(), l1.get_b(), expected_w, expected_b)


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

    first_err = learn.calculate_error(input, output)
    learn.run_epoch(input, output, batch_size=2, shuffle=False)

    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

    expected_b = np.array([0.105, 0.19, 0.25])
    expected_w = np.array(
        [
            [0.395, 0.7025],
            [-.49, .795],
            [.575, -.85]
        ]
    )
    assest_weight_bias(l.get_w(), l.get_b(), expected_w, expected_b)


def test_softmax():
    l = Layer(
        activation="softmax",
        bias=np.array([.1, .2]),
        weight=np.array([
            [.4, .7],
            [-.5, .8]
        ]),
        is_output=True
    )

    nn = NeuralNetwork()
    nn.add_layer(l)

    input = np.array([[-1., .5], [.5, -1.]])
    output = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )

    learn = NeuralNetworkLearning(nn, learning_rate=0.1)

    first_err = learn.calculate_error(input, output)
    learn.run_epoch(input, output, batch_size=2, shuffle=False)

    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

    expected_b = np.array([0.11301357, 0.18698643])
    expected_w = np.array(
        [
            [0.29539055, 0.79810267],
            [-0.39539055, 0.70189733]
        ]
    )

    assest_weight_bias(l.get_w(), l.get_b(), expected_w, expected_b)


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
    first_err = learn.calculate_error(input, output)

    learn.run_epoch(input, output, batch_size=2, shuffle=False)
    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

    expected_b = np.array([0.1012, 0.3006, 0.1991])
    expected_w = np.array(
        [
            [0.4024, 0.1018],
            [0.201, -0.799],
            [-0.7019, 0.4987]
        ]
    )

    assest_weight_bias(l1.get_w(), l1.get_b(), expected_w, expected_b)


def test_sgd_sigmoid():
    l = Layer(
        activation="sigmoid",
        bias=np.array([.15, .25]),
        weight=np.array([
            [.2, .35],
            [.3, .35]
        ]),
        is_output=True
    )

    input = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
    )
    output = np.array(
        [
            [0.1, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0]
        ]
    )

    nn = NeuralNetwork()
    nn.add_layer(l)

    learn = NeuralNetworkLearning(nn, learning_rate=0.1, update_bias=True)
    err0 = learn.calculate_error(input, output)

    for _ in range(10_000):
        err = learn.calculate_error(input, output)
        learn.run_epoch(input, output, batch_size=2)

    err = learn.calculate_error(input, output)

    assert err0 > err
    assert err < 0.2


def test_linear_2():
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

    learn = NeuralNetworkLearning(nn, learning_rate=0.1)
    first_err = learn.calculate_error(input, output)

    for _ in range(1):
        learn.run_epoch(input, output, batch_size=2, shuffle=False)

    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

    expected_b = np.array([0.22, 0.36, 0.11])
    expected_w = np.array(
        [
            [0.64, 0.28],
            [0.3, -0.7],
            [-0.89, 0.37]
        ]
    )

    assest_weight_bias(l1.get_w(), l1.get_b(), expected_w, expected_b)


@pytest.mark.skip(reason="TC masih ketabrak")
def test_mlp():
    l1 = Layer(
        activation="linear",
        is_output=False,
        bias=np.array([0.1, 0.2]),
        weight=np.array([
            [0.4, 0.7],
            [-0.5, 0.8]
        ])
    )
    l2 = Layer(
        activation="relu",
        is_output=True,
        bias=np.array([0.1, 0.2]),
        weight=np.array([
            [0.4, 0.7],
            [-0.5, 0.8]
        ])
    )

    nn = NeuralNetwork()
    nn.add_layer(l1)
    nn.add_layer(l2)

    input = np.array([
        [-1.0, 0.5],
        [0.5, -1.0]
    ])
    output = np.array([
        [0.1, 1.0],
        [1.0, 0.1]
    ])

    learn = NeuralNetworkLearning(nn, learning_rate=0.1)
    first_err = learn.calculate_error(input, output)

    learn.run_epoch(input, output, batch_size=2)
    last_err = learn.calculate_error(input, output)

    # Apakah error berkurang?
    assert last_err - first_err < 0

    # Testing
    expected_weight_l2 = np.array([
        [0.35605, 0.5281],
        [-0.504275, 0.78545]
    ])
    expected_new_b2 = np.array([0.121, 0.2045])

    expected_weight_l1 = np.array([
        [0.42885, 0.685575],
        [-0.4403, 0.77015]
    ])
    expected_new_b1 = np.array([0.07115, 0.1403])

    assest_weight_bias(l1.get_w(), l1.get_b(),
                       expected_weight_l1, expected_new_b1)
    assest_weight_bias(l2.get_w(), l2.get_b(),
                       expected_weight_l2, expected_new_b2)
