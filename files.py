import pickle


def save_file(data, path):
    file = open(path, "wb")
    pickle.dump(data, file)
    file.close()

def load_file(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data