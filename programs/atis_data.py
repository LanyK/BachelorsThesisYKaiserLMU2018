# import the ATIS Corpus

import json, sys, keras
import numpy as np

def toWord(index):
    return index_to_word[index]

def toLabel(index):
    return index_to_label[index]

index_to_word = None
index_to_label = None

def load_atis_data():
    # Load and Prepare Corpus data
    print("Loading input data...", end="", flush=True)

    atis_filename = "atis.json"
    atis = json.load(open(atis_filename, "r"))

    ## Create Sets (Eval,...) and Mappings
    word_to_index = atis["vocab"]
    index_to_word = {val:key for key,val in word_to_index.items()}
    train_sents = atis["train_sents"]
    test_sents = atis["test_sents"]
    train_labels = atis["train_labels"]
    test_labels = atis["test_labels"]
    label_to_index = atis["label_dict"]
    index_to_label = {val:key for key,val in label_to_index.items()}

    ## Prepare Data input (Numpy Arrays)
    vocab_size = len(word_to_index)
    num_classes = max([max(sent) for sent in train_labels]) + 1 if min([min(sent) for sent in train_labels]) == 0 else 0
    max_sent_len = max([len(t) for t in train_sents] + [len(t) for t in test_sents])

    print("DONE")

    ## Format the model input data as sliding windows
    print("Preparing Sliding Window Data...", end="", flush=True)

    PADDING = 572
    WINDOW_SIZE = 9

    test_x = []
    train_x = []

    for sent in test_sents:
        for i in range(len(sent)):
            window = []
            for j in range(-4,5):
                k = i + j
                if k < 0 or k >= len(sent):
                    window.append(PADDING)
                else:
                    window.append(sent[k])
            test_x.append(window)

    for sent in train_sents:
        for i in range(len(sent)):
            window = []
            for j in range(-4,5):
                k = i + j
                if k < 0 or k >= len(sent):
                    window.append(PADDING)
                else:
                    window.append(sent[k])
            train_x.append(window)

    np_test_x = np.array(test_x)
    np_train_x = np.array(train_x)

    np_train_y = np.reshape(np.array([x for sent in train_labels for x in sent]), -1)
    np_train_y = keras.utils.to_categorical(np_train_y, num_classes)
    np_test_y = np.reshape(np.array([x for sent in test_labels for x in sent]), -1)
    np_test_y = keras.utils.to_categorical(np_test_y, num_classes)

    print("DONE")

    return (np_train_x, np_train_y, np_test_x, np_test_y, vocab_size, index_to_word, WINDOW_SIZE)

if __name__ == "__main__":
    pass
