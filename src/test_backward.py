from Layer import *


def test_sigmoid_backpropagation():
    """
    Test sigmoid backpropagation.

    Calculation Source : 
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    """
    wl1 = np.array([
        [0.15, 0.20],
        [0.25, 0.30]
    ])
    wl2 = np.array([
        [0.40, 0.45],
        [0.50, 0.55]
    ])
    b1 = np.array([0.35, 0.35])
    b2 = np.array([0.6, 0.6])

    l1 = Layer(
        activation="sigmoid",
        bias=b1,
        is_output=False,
        weight=wl1
    )

    l2 = Layer(
        activation="sigmoid",
        bias=b2,
        weight=wl2,
        is_output=True
    )

    input = np.array([[0.05, 0.1]])
    output = np.array([[0.01, 0.99]])

    res_l1 = l1.calculate(input)
    res_l2 = l2.calculate(res_l1)

    # BACKWARD PROPS
    learning_rate = 0.5

    # Layer 2
    delta_l2 = l2.layer_delta(res_l1, expected=output)
    weight_delta_l2, bias_delta_l2 = l2.weight_diff(res_l1, delta_l2)

    new_weight_l2 = wl2 - learning_rate * weight_delta_l2

    expected_weight_l2 = np.array([
        [0.35891648, 0.408666186],
        [0.511301270, 0.561370121]
    ])

    sse = np.square(expected_weight_l2-new_weight_l2).sum()
    assert sse < 10e-6

    # Layer 1
    error_diff = l2.error_diff(delta_l2)
    delta_l1 = l1.layer_delta(input, error_diff=error_diff)
    weight_delta_l1, bias_delta_l1 = l1.weight_diff(input, delta_l1)

    new_weight_l1 = wl1 - learning_rate * weight_delta_l1

    expected_weight_l1 = np.array([
        [0.149780716, 0.19956143],
        [0.24975114, 0.29950229]
    ])

    sse = np.square(expected_weight_l1-new_weight_l1).sum()
    assert sse < 10e-6
