from NeuralNetworkFactory import *
import os.path as path


def main():
    f = NeuralNetworkFactory()
    nn = f.assistant_json(
        path.join(path.dirname(__file__),
                  "../test/sample/assistant/multilayer.json")
    )
    nn.draw()

    res = nn.calculate([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    print(res)


if __name__ == "__main__":
    main()
