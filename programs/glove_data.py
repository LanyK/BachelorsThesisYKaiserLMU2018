import sys
import numpy as np

def load_glove_data(emb_size, vocab_size, index_to_word_dict):
    # Load and Prepare Glove data
    print("Loading glove data...", flush=True, end="")
    all_embeddings = {}

    glove_filename = "glove\\glove.6B." + str(emb_size) + "d.txt"
    l = 1
    with open(glove_filename,"r",encoding="UTF8") as glovedata:
        for line in glovedata:
            sys.stdout.write("\rLoading glove data..." + str(l))
            sys.stdout.flush()
            l+= 1
            word, *emb = line.rstrip().split()
            all_embeddings[word] = np.asarray(emb, dtype='float32')

    sys.stdout.write("\rLoading glove data...DONE     \n")
    sys.stdout.flush()

    embedding_matrix = np.zeros((vocab_size, emb_size))
    for index, word in index_to_word_dict.items():
        emb = all_embeddings.get(word)
        if emb is not None:
            embedding_matrix[index] = emb
        else:
            print(" WARNING:",word,"is unknown to pretrained embeddings.")

    del all_embeddings
    return embedding_matrix

if __name__ == "__main__":
    pass
