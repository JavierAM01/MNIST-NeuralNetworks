from matplotlib import pyplot as plt
import torch as T
import os

from nn import Model
from files import save_file, load_file



def load_and_save_data():

    from keras.datasets import mnist
    (X, Y), (test_X, test_y) = mnist.load_data()
    
    save_file(X, "data/train_X")
    save_file(Y, "data/train_y")
    save_file(test_X, "data/test_X")
    save_file(test_y, "data/test_y")

    return X, Y

def create_model():

    # load data MNIST
    if not os.path.exists("data/test_X"):
        X, Y = load_and_save_data()
    else:
        X, Y = load_file("data/train_X"), load_file("data/train_y")

    f = lambda n : [int(i==n) for i in range(10)]
    X = T.tensor(X, dtype=T.float).unsqueeze(1)
    Y = T.tensor(list(map(f, Y)), dtype=T.float)

    # ask for parameters
    print("\nPath to save the model:")
    save_path = input(" > ")
    print("\nNumber of epochs:")
    epochs = int(input(" > "))
    print("\nBatch size:")
    batch_size = int(input(" > "))

    # create & train model
    nn = Model(0.001)
    history = nn.fit(X,Y, epochs=epochs, batch_size=batch_size)
        
    # plot results
    k = len(history) // 100

    y = history[::k]
    x = range(len(y))

    plt.plot(x, y)
    plt.title("Training loss")

    # save results
    if not os.path.exists(f"models/{save_path}"):
        os.mkdir(f"models/{save_path}")

    plt.savefig(f"models/{save_path}/losses.jpg", dpi=200, bbox_inches='tight')
    nn.save_model(f"models/{save_path}/model.pkl")

    # show plot
    plt.show()





