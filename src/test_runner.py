import os.path as path
from NeuralNetworkFactory import *
import os

ASSISTANT_PATH = path.join(path.dirname(__file__), "../sample/assistant")


def test_tc():
    for root, _, files in os.walk(ASSISTANT_PATH):
        for i in files:
            try:
                if i.endswith(".json"):
                    file_path = path.join(root, i)

                    with open(file_path, 'r') as f:
                        json_info = json.load(f)

                    nn = NeuralNetworkFactory().assistantJson(file_path)

                    expect = json_info["expect"]

                    sse_threshold = expect["max_sse"]
                    answer_output = np.array(expect["output"])

                    res = nn.calculate(np.array(json_info["case"]["input"]))

                    sse_actual = np.square(
                        res - answer_output).sum(axis=0) / 2

                    assert np.all(sse_actual <= sse_threshold)
            except Exception as err:
                print(f"Failed when execute file {i}")
                raise err
