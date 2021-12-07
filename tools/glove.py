import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    embeddings_dict = {}
    filename = '../data/glove/glove.6B.300d.txt'
    print("generate word list...")
    with open(filename, 'r', encoding="utf-8") as f:
        print(f"load file \"{filename}\"")
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    np.save('../data/glove/wordsList', np.array(list(embeddings_dict.keys())))
    np.save('../data/glove/wordVectors',
            np.array(list(embeddings_dict.values()), dtype='float32'))
    print("word list saved")


if __name__ == '__main__':
    main()
