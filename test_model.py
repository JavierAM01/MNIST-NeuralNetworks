import torch as T
import numpy as np

from nn import Model
from files import load_file


def test_model():

    # load test data
    X_test, y_test = load_file("data/test_X"), load_file("data/test_y") 
    X_test = T.tensor(X_test, dtype=T.float).unsqueeze(1).unsqueeze(1)
    
    # load model
    print("\nModel path:")
    load_path = input(" > ")
    nn = Model(0.001)
    nn.load_model(f"models/{load_path}/model.pkl")

    # make predictions
    correct = 0
    for x, y in zip(X_test, y_test):
        pred = np.argmax(nn(x)[0].tolist())
        if pred == y:
            correct += 1
            
    print("Number of predictions:", correct, f" ({100*correct/len(X_test)} %)")
