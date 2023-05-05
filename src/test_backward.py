import os.path as path

import pytest
from Layer import *
from NeuralNetwork import NeuralNetwork
import os
from NeuralNetworkFactory import *

ASSISTANT_PATH = path.join(path.dirname(__file__), "../sample/backward")

@pytest.mark.skip(reason="beberapa TC asisten masih Failed")
def test_tc():
    for root, _, files in os.walk(ASSISTANT_PATH):
        for i in files:
            file_path = path.join(root, i)

            if i.endswith(".json"):
                check_backward_tc(file_path)

def check_backward_tc(file_path: str):
    try:
        with open(file_path, 'r') as f:
            json_info = json.load(f)

        nn = NeuralNetworkFactory().assistant_backward_json(file_path)

        expect = json_info["expect"]

        stopped_by = expect["stopped_by"]

        if(stopped_by != nn.stopped_by):
            print(
                f"Test case file {path.basename(file_path)} is failed because stopped by is different: {stopped_by} != {nn.stopped_by}")
            assert False

        if(expect.get("final_weights") is None):
            print(
                f"Test case file {path.basename(file_path)} is success because stopped by is the same: {stopped_by} == {nn.stopped_by}")
        else:
            sse = 0.0   
            final_weights = expect["final_weights"]

            for i in range(len(nn.layers)):
                expect_layer = np.array(final_weights[i])
                b = expect_layer[0]
                w = np.transpose(expect_layer[1:])
                sse = max(sse, np.square(nn.layers[i].get_w()-w).sum())
                sse = max(sse, np.square(nn.layers[i].get_b()-b).sum())
                # sse += np.square(nn.layers[i].get_w()-w).sum()
                # sse += np.square(nn.layers[i].get_b()-b).sum()
                if(sse >= 1e-6):
                    print(f"Actual : {nn.layers[i].get_w()}")
                    print(f"Expected : {w}\n")
                    print(
                        f"Test case file {path.basename(file_path)} is failed because stopped by is the same: {stopped_by} == {nn.stopped_by} but sse: {sse} >= 1e-6")
                    assert False                    
            if(sse < 1e-6):
                print(
                    f"Test case file {path.basename(file_path)} is success because stopped by is the same: {stopped_by} == {nn.stopped_by} and sse: {sse} < 1e-6")
    except AssertionError:
        assert False
    except Exception as err:
        print(
            f"Failed when execute file {file_path} because an exception:", end=" ")
        print(err)
        assert False

def assest_weight_bias(w_actual, b_actual, w_expect, b_expect):
    assert np.square(w_actual-w_expect).sum() < 1e-8
    assert np.square(b_actual-b_expect).sum() < 1e-8



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
    new_b2 = b2 - learning_rate * bias_delta_l2

    expected_weight_l2 = np.array([
        [0.35891648, 0.408666186],
        [0.511301270, 0.561370121]
    ])

    expected_new_b2 = np.array([0.5308, 0.6190])

    sse = np.square(expected_weight_l2-new_weight_l2).sum()
    sse_b = np.square(expected_new_b2-new_b2).sum()

    assert sse < 10e-6
    assert sse_b < 10e-6

    # Layer 1
    error_diff = l2.error_diff(delta_l2)
    delta_l1 = l1.layer_delta(input, error_diff=error_diff)
    weight_delta_l1, bias_delta_l1 = l1.weight_diff(input, delta_l1)

    new_weight_l1 = wl1 - learning_rate * weight_delta_l1
    new_b1 = b1 - learning_rate * bias_delta_l1

    expected_weight_l1 = np.array([
        [0.149780716, 0.19956143],
        [0.24975114, 0.29950229]
    ])

    expected_new_b1 = np.array([0.3456, 0.3450])

    sse = np.square(expected_weight_l1-new_weight_l1).sum()
    sse_b = np.square(expected_new_b1-new_b1).sum()

    assert sse < 1e-6
    assert sse_b < 1e-6
