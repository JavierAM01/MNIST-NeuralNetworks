from create_model import create_model
from test_model import test_model



def main():
    print("Chose an option:")
    print(" 1) Train model")
    print(" 2) Test model")
    x = input(" > ")

    if x == "1":
        create_model()
    else:
        test_model()




if __name__ == "__main__":
    main()