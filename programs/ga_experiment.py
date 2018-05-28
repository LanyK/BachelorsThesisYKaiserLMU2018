# import the ATIS Corpus

import sys, keras, time, math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras import backend as kerasBackend
from collections import defaultdict
from traceback import format_exc
import gc
import ga_lib, atis_data, glove_data

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def one_float():
    return 1.0

# Global settings. TODO these should probably be put into a settings file.
population_size = 20
training_epochs = 20
generations = 10
learning_rate = 3
batch_size = 16
decoder = None
train_x, train_y, test_x, test_y, vocab_size, window_size = (None,None,None,None,None,None)
embeddings = None
eval_biases = [3,2,1] # performance, size, overfitting
min_size_train_time = None
max_size_train_time = None
class_weights = defaultdict(one_float)
test_y_class_counts = []
train_y_class_counts = []

class F1History(keras.callbacks.Callback):
    def __init__(self, train_data):
        # self.sample_weight = np.array([0.0] + [1.0 for _ in range(126)])
        self.train_data = train_data

        global test_y_class_counts
        self.test_y_zero_indexes = []
        for i in range(len(test_y_class_counts)):
            if test_y_class_counts[i] == 0 and i > 0: # dont remove the 0 tag, it will be dealt with later.
                self.test_y_zero_indexes.append(i)

        global train_y_class_counts
        self.train_y_zero_indexes = []
        for i in range(len(train_y_class_counts)):
            if train_y_class_counts[i] == 0 and i > 0: # dont remove the 0 tag, it will be dealt with later.
                self.train_y_zero_indexes.append(i)

    def on_train_begin(self, logs={}):
        self.trainf1s = []
        self.evalf1s = []

    def on_epoch_end(self, epoch, logs={}):
        # Eval Data
        predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        targ = self.validation_data[1]

        recall = recall_score(targ, predict, average=None)
        recall = [value for i,value in enumerate(recall) if i not in self.test_y_zero_indexes] # remove faulty entries (sklearn returns 0 for labels not in the gold standard)
        recall = np.average(recall[:-1]) # Ignore Class '0'
        precision = precision_score(targ, predict, average=None)
        precision = [value for i,value in enumerate(precision) if i not in self.test_y_zero_indexes] # remove faulty entries (sklearn returns 0 for labels not in the gold standard)
        precision = np.average(precision[:-1]) # Ignore Class '0'
        f1 = 2 * (precision * recall) / (precision + recall)
        self.evalf1s.append(f1)
        print("Eval. Data - F1 Score:", f1, "Precision:",precision,"Recall:",recall)

        # Train Data
        predict = (np.asarray(self.model.predict(self.train_data[0]))).round()
        targ = self.train_data[1]

        recall = recall_score(targ, predict, average=None)
        recall = [value for i,value in enumerate(recall) if i not in self.train_y_zero_indexes] # remove faulty entries (sklearn returns 0 for labels not in the gold standard)
        recall = np.average(recall[:-1]) # Ignore Class '0'
        precision = precision_score(targ, predict, average=None)
        precision = [value for i,value in enumerate(precision) if i not in self.train_y_zero_indexes] # remove faulty entries (sklearn returns 0 for labels not in the gold standard)
        precision = np.average(precision[:-1]) # Ignore Class '0'
        f1 = 2 * (precision * recall) / (precision + recall)
        self.trainf1s.append(f1)

        global training_epochs
        if epoch + 1 == training_epochs:
            print("Train Data - F1 Score:", f1, "Precision:",precision,"Recall:",recall)

class KerasCNNGenomeDecoder():
    """ Class to build Convolutional Neural Networks from TODO.CNNGenome instances.
    """
    def __init__(self):
        pass

    def decode(self, cnn_genome, vocab_size, window_size):
        """ Retuns a Keras model decoded from the genome
        """
        cnn_layers = cnn_genome.get_conv_layer_count()
        dense_layers = cnn_genome.get_dense_layer_count()
        conv_attributes = cnn_genome.get_conv_layer_attributes
        dense_attributes = cnn_genome.get_dense_layer_attributes
        trainable_values_count = 0
        global embeddings

        model = Sequential()
        model.add(Embedding(vocab_size, 200, input_length=window_size, weights=[embeddings], trainable=False))

        print("Decoding:")
        for i in range(1,cnn_layers + 1):
            print(" conv: ",conv_attributes(i)[0],(conv_attributes(i)[1],),conv_attributes(i)[2])
            conv = Conv1D(conv_attributes(i)[2], kernel_size=(conv_attributes(i)[0],), strides=conv_attributes(i)[1], activation='tanh')
            model.add(conv)
            trainable_values_count += conv.count_params()
            model.add(MaxPooling1D(2, padding="same"))
        model.add(Flatten())

        for i in range(1,dense_layers + 1):
            print(" dense: ",dense_attributes(i))
            dense = Dense(dense_attributes(i)[0], activation='tanh')
            model.add(dense)
            trainable_values_count += dense.count_params()
            model.add(Dropout((dense_attributes(i)[1] * 0.05)))
        dense = Dense(127, activation='softmax')
        model.add(dense)
        trainable_values_count += dense.count_params()

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return (model, trainable_values_count)

def compute_onehot_class_counts(y):
    counts = [0 for x in y[0]]
    for onehot in y:
        for i in range(0,onehot.size):
            if onehot[i] == 1:
                counts[i] += 1
                break
    return counts

def compute_class_weights(num_classes, *y_sets):
    """ Distribute num_classes probability mass
        over the classes according to their inverse
        relative rarity
    """
    class_weights = defaultdict(one_float)
    global learning_rate
    counts = [0 for _ in range(num_classes)]
    for y_set in y_sets:
        for y_arr in y_set:
            y = np.argmax(y_arr)
            counts[y] += 1
    total = sum(counts)
    avg = total / num_classes
    raw_weight_sum = sum([total / counts[index] for index in range(num_classes)])
    weights = [(total * num_classes ) / (counts[i] * raw_weight_sum) for i in range(num_classes)]

    for i in range(num_classes):
        class_weights[i] = weights[i] * learning_rate

    return class_weights


def eval(f1, network_size, eval_f1):
    perf_score = convert_to_score(eval_f1, algo="performance") * eval_biases[0]
    size_score = convert_to_score(network_size, algo="network_size") * eval_biases[1]
    f1_diff = eval_f1 - f1
    overfitting_score = convert_to_score(f1_diff, algo="overfitting") * eval_biases[2]
    print("EVALUATION: Perf:",perf_score,"Size:",size_score,"Overfit:",overfitting_score,"(",f1, network_size, eval_f1,")")
    # failsafe, because scipy.
    if math.isnan(perf_score):
        perf_score = 1
    if math.isnan(size_score):
        size_score = 1
    if math.isnan(overfitting_score):
        overfitting_score = 1
    return (perf_score + size_score + overfitting_score) / 3.0

def convert_to_score(value, algo="none"):
    """ algo can be one of ["none", "network_size", "performance", "overfitting"]
        value should be:
            network_size: total number of weight values in the network
            performance: F1 score
            overfitting: eval_F1 - train_F1
    """
    if algo == "none":
        return value
    elif algo == "network_size":
        global max_network_size
        global min_network_size
        return max(100 - 100 * ((max(1,value - min_network_size) / (max_network_size - min_network_size))),1)
    elif algo == "overfitting":
        if value > 0:
            return 100
        else:
            return max(1,100.0+((value*100)*(10/3))) # up to 0.3
    elif algo == "performance":
        return value**2 * 100
    else:
        raise ValueError("Invalid algo given.")

def build_and_eval_cnn(cnn_genome):
    global batch_size
    global class_weights
    global train_x
    global train_y
    result = (0,0)
    try:
        model, trainable_values_count = decoder.decode(cnn_genome, vocab_size, window_size)
        network_size = trainable_values_count

        # model.summary()
        # time_callback = TimeHistory()
        f1_callback = F1History((train_x,train_y))
        model.fit(train_x, train_y, epochs=training_epochs, class_weight=class_weights, batch_size=batch_size, callbacks=[f1_callback], verbose=1, validation_data=(test_x, test_y))

        eval_f1 = f1_callback.evalf1s[-1]
        train_f1 = f1_callback.trainf1s[-1]

        print("NetworkSize:",network_size,"Train F1:",train_f1,"Eval F1:",eval_f1)
        # result = model.evaluate(test_x, test_y, batch_size=None, verbose=1, sample_weight=None, steps=None)
    except Exception as e:
        print(e)
        print("!!! impossible structure. ", cnn_genome)
        raise e
        # raise
    return eval(train_f1, network_size, eval_f1)

def main():
    global population_size
    global batch_size
    global generations
    global training_epochs
    global train_x
    global train_y
    global test_x
    global test_y
    global vocab_size
    global window_size
    train_x, train_y, test_x, test_y, vocab_size, index_to_word, window_size = atis_data.load_atis_data()

    global embeddings
    embeddings = glove_data.load_glove_data(200, vocab_size, index_to_word)
    del index_to_word

    # compute the class weight from the obtained counts...
    global class_weights
    class_weights = compute_class_weights(np.argmax(train_y) + 1, train_y, test_y)

    # Precompute class counts for train and test data to avoid sklearn shenanigans later down the line
    global test_y_class_counts
    test_y_class_counts = compute_onehot_class_counts(test_y)

    global train_y_class_counts
    train_y_class_counts = compute_onehot_class_counts(test_y)

    # Prepare the decoder
    global decoder
    decoder = KerasCNNGenomeDecoder()

    # Get the min network size
    global min_network_size
    min_network_size = 0
    min_size_genome = ga_lib.CNNGenome(window_size); min_size_genome.set_to_min_size()
    model, trainable_values_count = decoder.decode(min_size_genome, vocab_size, window_size)
    model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1, class_weight=class_weights, validation_data=(test_x, test_y))
    min_network_size = trainable_values_count
    print("Min Network Size:", min_network_size)

    # Get the max network size
    global max_network_size
    max_network_size = 0
    max_size_genome = ga_lib.CNNGenome(window_size); max_size_genome.set_to_max_size()
    model, trainable_values_count = decoder.decode(max_size_genome, vocab_size, window_size)
    model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1, class_weight=class_weights, validation_data=(test_x, test_y))
    max_network_size = trainable_values_count
    print("Max Network Size:", max_network_size)

    # ----------------------- GA -----------------------------------------------
    #
    # --------------------------------------------------------------------------
    population = ga_lib.Population(population_size, ga_lib.CNNGenome, window_size, mutation_prob = 0.05)
    population.random_fill()
    population.set_fitness_function(build_and_eval_cnn)

    population.evaluate_generation()
    print("Gen",0,"----------------------------")
    print("Population:",population)
    print("Base Fitness avg:",population.get_avg_fitness())

    for i in range(1,generations+1):
        population.advance_one_generation()
        population.evaluate_generation()
        kerasBackend.clear_session() # prevent clogging up of things
        gc.collect()
        print("Gen",i,"----------------------------")
        print("Population:",population)
        print("Fitness avg:",population.get_avg_fitness())
        # print(population)

    with open("generation_history.txt", "w") as logfile:
        for gen_no, pop in population.get_generation_archive().items():
            fitnesses = []
            logfile.write(str(gen_no) + ":\n")
            logfile.write("BEST_SCORE:\t" + str(pop[0][1]) + "\n")
            for genome, fitness in pop:
                logfile.write("  " + str(fitness) + "\t" + str(genome) + "\n")
                fitnesses.append(fitness)
            logfile.write("AVG_SCORE:\t" + str(sum(fitnesses) / float(len(fitnesses))) + "\n")
        logfile.write("LATEST:\n")
        logfile.write("BEST_SCORE:\t" + str(population.pop[0][1]) + "\n")
        for genome, fitness in population.pop:
            logfile.write("  " + str(fitness) + "\t" + str(genome) + "\n")
            fitnesses.append(fitness)
        logfile.write("AVG_SCORE:\t" + str(population.get_avg_fitness()))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("log.txt", "a") as lf:
            lf.write("\n")
            lf.write(format_exc())
            raise e
