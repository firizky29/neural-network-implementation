import os.path as path
from NeuralNetworkFactory import *
import os

ASSISTANT_PATH = path.join(path.dirname(__file__), "../sample/forward")


def test_tc():
    for root, _, files in os.walk(ASSISTANT_PATH):
        for i in files:
            file_path = path.join(root, i)

            if i.endswith(".json"):
                check_sse(file_path)


def main():
    for root, _, files in os.walk(ASSISTANT_PATH):
        for i in files:
            try:
                file_path = path.join(root, i)

                if i.endswith(".json"):
                    check_sse(file_path)
            except AssertionError:
                pass


def check_sse(file_path: str):
    try:
        with open(file_path, 'r') as f:
            json_info = json.load(f)

        nn = NeuralNetworkFactory().assistantJson(file_path)

        expect = json_info["expect"]

        sse_threshold = expect["max_sse"]
        answer_output = np.array(expect["output"])

        res = nn.calculate(np.array(json_info["case"]["input"]))

        sse_actual = np.square(
            res - answer_output).sum(axis=0)

        if res.shape != answer_output.shape:
            print(
                f"Test case file {path.basename(file_path)} is failed because shape is different: {res.shape} != {answer_output.shape}")
            print(f"Actual : {res}")
            print(f"Expected : {answer_output}\n")
            assert False

        if not np.all(sse_actual <= sse_threshold):
            print(
                f"Test case file {path.basename(file_path)} is failed because SSE is greater than threshold: {sse_actual} > {sse_threshold}")
            print(f"Actual : {res}")
            print(f"Expected : {answer_output}\n")
            assert False

        print(f"Test case file {path.basename(file_path)} is success")
    except AssertionError:
        assert False
    except Exception as err:
        print(
            f"Failed when execute file {file_path} because an exception:", end=" ")
        print(err)
        assert False


if __name__ == "__main__":
    main()
