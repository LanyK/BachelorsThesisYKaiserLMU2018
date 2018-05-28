import sys, random, math
import matplotlib.pyplot as plt

# Global variables. Tuned for the ATIS task. TODO Should be put into an external config file.
default_max_conv_layers = 2
default_max_dense_layers = 3
cnn_input_length = 9
max_mutation_attempts = 5

class Population():
    """ Represents a population of individuals for Genetic Algorithms.
        The individuals are provided as a genome class supporting the following operations:
            - self.mutate()          -> applying random mutation to a gene
            - self.crossover(other)  -> creating an offspring from self and other
            - Class.create_random()  -> creating a new random genome
        A fitness_function can be provided to make the population handle the rating internally.
    """
    def __init__(self, size, GenomeClass, network_input_dimensionality, mutation_prob = 0.05):
        self.popsize = size
        self.genome_class = GenomeClass
        self.pop = [(None,None) for i in range(self.popsize)]
        self.fitness_function = lambda x : raise_(RuntimeError("No fitness function set!"))
        self.generation_archive = {}
        self.generation_no = 0
        self.mutation_prob = mutation_prob
        self.network_input_dimensionality = network_input_dimensionality

    def advance_one_generation(self, selection_mode='roulette'):
        """ selection_mode has to be one of 'roulette' or 'tournament'. Will default to 'roulette'.
            Either way, the best individual will be kept (1-individual-elitism)

            Tournament Selection:
                For each individual:
                    pick random other individual
                    compare fitness
                    put winner in mating pool
                pick 2 random individuals out of the pool to mate

            Roulette Selection:
                Pick 2 individuals to mate where the probability is fitness / sum of fitnesses

            If this is applied to a not yet rated population, it will try to apply its provided fitness_function.
            Mutation is applied after crossover.
        """
        if not self.is_rated():
            self.evaluate_generation()

        if selection_mode not in ['tournament','roulette']:
            selection_mode = "roulette"

        old_pop = self.pop

        if selection_mode == "roulette":
            print("<Fitnesses> ", [f for (x,f) in self.pop])
            new_pop = [(self.pop[0][0].clone(),None)]
            while len(new_pop) < self.popsize:
                father = weighted_choice(self.pop)
                mother = father
                while mother is father:
                    mother = weighted_choice(self.pop)
                offspring = father.crossover(mother)
                new_pop.append((offspring,None))
            self.pop = new_pop

        elif selection_mode == "tournament":
            new_pop = [(self.pop[0][0].clone(),None)] # Elitism
            mating_pool = []
            for individual,fitness in self.pop:
                other = individual
                while individual is other:
                    other, other_fitness = self.pop[random.randint(0,self.popsize - 1)]
                if fitness >= other_fitness:
                    mating_pool.append((individual,fitness))
                else:
                    mating_pool.append((other,other_fitness))
            while len(new_pop) < self.popsize:
                father = mating_pool[random.randint(0,self.popsize - 1)]
                mother = father
                while mother is father:
                    mother = mating_pool[random.randint(0,self.popsize - 1)]
                offspring = father.crossover(mother)
                new_pop.append((offspring,None))
            self.pop = new_pop
        else:
            raise ValueError("Unknown selection mode")

        self.generation_archive[self.generation_no] = [(ind,fit) for (ind,fit) in old_pop]
        self.generation_no += 1

        for (individual,fitness) in self.pop[1:]:
            individual.mutate(self.mutation_prob) # optional arguments?

    def get_avg_fitness(self):
        return sum((fitness for (_,fitness) in self.pop)) / float(self.popsize) if self.is_rated else None

    def set_mutation_prob(self, mutation_prob):
        self.mutation_prob = mutation_prob

    def random_fill(self):
        self.pop = [(self.genome_class(self.network_input_dimensionality, random=True),None) for i in range(self.popsize)]

    def set_fitness_function(self, Fitness_Function):
        """ This function is used when rate_individuals(do_sort_internally=True) is called
        """
        self.fitness_function = Fitness_Function

    def evaluate_generation(self):
        """ applies the provided fitness function and sorts the individuals. subsequent calls to get them will return a sorted list.
        """
        self.pop = sorted([(individual, self.fitness_function(individual)) for (individual,_) in self.pop], key=lambda x : x[1], reverse=True)


    def get_generation(self):
        """ Returns the individuals as a list of tuples (individual,fitness). If the individuals are rated, i.e.
            rate_individuals or set_fitnesses was called before, this list will be sorted by fitness.
            If no rating happened, the tuples will have the shape (individual, None).
            See also is_rated() to check up on this.
        """
        return self.pop

    def get_generation_archive(self):
        return self.generation_archive

    def is_rated(self):
        """ Returns true if the population has fitness scores evaluated for them.
        """
        return all(y != None for (x,y) in self.pop)

    def set_fitnesses(self, fitnesses):
        """ Set the fitness scores from an external source. This method is to be used when
            you do not want to use the providable fitness_function.
            The population will be internally sorted by fitness after this call.
        """
        if len(fitnesses) < self.popsize:
            raise ValueError("Popsize is", self.popsize, "! fitnesses of length", len(fitnesses), "given.")
        self.pop = sorted([(self.pop[i], fitnesses[i]) for i in range(len(self.pop))], key=lambda x : x[1], reverse=True)

    def print_generation(self):
        print(self.pop)

    def __str__(self):
        return str(self.pop)

    def __repr__(self):
        return "<Population with active generaton: " + self.__str__() + ", size " + str(self.popsize) + ">"

class CNNGenome():
    """ A genome(as integer list) that can be decoded to a Keras
    Convolutional Neural Nework via the CNNGenomeDecoder class

    CNNGenomes encode a CNN Architecture of A layers of combined Convolution1D-MaxPooling
    and B hidden densly connected layers with dropout on top of the convolution.
    Flattening, Embedding Input and the Output Layer are handeled by the Decoder.

    Genome shape:  AB(kernel_size, strides, filter_amount)){A}(node_count, dropout_factor){B}
    Example: [3,2,3,1,256,4,1,128,3,1,128,2500,1500]
    """
    def __init__(self, network_input_dimensionality, random=False, max_conv_layers=-1, max_dense_layers=-1):
        self.network_input_dimensionality = network_input_dimensionality
        self.genome = []
        if random:
            mx_conv_layers = max_conv_layers if max_conv_layers > 0 else default_max_conv_layers
            mx_dense_layers = max_dense_layers if max_dense_layers > 0 else default_max_dense_layers
            self.randomize(mx_conv_layers, mx_dense_layers)

    def set_genome(self,new_genome = []):
        if CNNGenome.validate_genome(new_genome) == False:
            raise ValueError("'" + str(new_genome) + "' is not a valid genome shape!")
        self.genome = new_genome

    def mutate(self, mutation_prob, max_conv_layers=-1, max_dense_layers=-1):
        """ Mutates this genome.
            The mutation probability is applied individually to each value in the genome.
            Then, with mutation_prob chance, the length of the genome is altered.
        """
        if max_conv_layers < 1:
            max_conv_layers = default_max_conv_layers
        if max_dense_layers < 1:
            max_dense_layers = default_max_dense_layers

        A = self.genome[0]
        B = self.genome[1]
        a = A
        a_data = self.genome[2:(3 * A) + 2]
        b = B
        b_data = self.genome[(3 * A) + 2:]

        # Length mutation
        ## Convolution layers
        len_change = "same"
        if random.random() < (mutation_prob/2):
            len_change = "grow" if random.random() < 0.5 else "shrink"

        if len_change == "grow":
            if a < max_conv_layers:
                a += 1
                try:
                    a_data += CNNGenome.generate_random_conv_layer_attributes(a, a_data, self.network_input_dimensionality)
                except Exception as e:
                    a -= 1
        elif len_change == "shrink":
            if a > 1:
                a -= 1
        else:
            pass

        ## Dense Layers
        len_change = "same"
        if random.random() < (mutation_prob/2):
            len_change = "grow" if random.random() < 0.5 else "shrink"

        if len_change == "grow":
            if b < max_dense_layers:
                b += 1
                b_data += CNNGenome.generate_random_dense_layer_attributes()
        elif len_change == "shrink":
            if b > 1:
                b -= 1
                b_data = b_data[:-2]
        else:
            pass

        # Values mutation
        ## Convolution layers
        for i in range(len(a_data)):
            if random.random() < mutation_prob:
                if i % 3 == 0: #kernel
                    old_value = a_data[i]
                    layer_no = math.floor(i/3.0) + 1
                    for m in range(max_mutation_attempts):
                        new_layer_information = CNNGenome.generate_random_conv_layer_attributes(layer_no, a_data[:(layer_no-1) * 3], self.network_input_dimensionality)
                        proposed_a_data = a_data[:]
                        proposed_a_data[i] = new_layer_information[0] # insert new kernel_size
                        if CNNGenome.validate_convolutional_genome_information(proposed_a_data, self.network_input_dimensionality):
                            a_data = proposed_a_data
                            break
                        else:
                            pass
                elif i % 3 == 1: #strides
                    old_value = a_data[i]
                    layer_no = math.floor(i/3.0) + 1
                    for m in range(max_mutation_attempts):
                        new_layer_information = CNNGenome.generate_random_conv_layer_attributes(layer_no, a_data[:(layer_no-1) * 3], self.network_input_dimensionality)
                        proposed_a_data = a_data[:]
                        proposed_a_data[i] = new_layer_information[1] # insert new kernel_size
                        if CNNGenome.validate_convolutional_genome_information(proposed_a_data, self.network_input_dimensionality):
                            a_data = proposed_a_data
                            break
                else: # i % 3 == 2 #filters
                    a_data[i] = CNNGenome.random_filter_count()
        ## Dense layers
        for i in range(len(b_data)):
            if random.random() < mutation_prob:
                if i % 2 == 0: # Neurons
                    b_data[i] = CNNGenome.random_neuron_count()
                else: # Dropout
                    b_data[i] = CNNGenome.random_dropout_factor()

        self.set_genome([a] + [b] + a_data + b_data)

    def crossover(self, other):
        """ Computes an offsprig genome out of the two parent genomes """
        finished = False
        while finished == False:
            self_A = self.genome[0]
            self_B = self.genome[1]
            self_A_data = self.genome[2:(3 * self_A) + 2]
            self_B_data = self.genome[(3 * self_A) + 2:]

            other_A = other.genome[0]
            other_B = other.genome[1]
            other_A_data = other.genome[2:(3 * other_A) + 2]
            other_B_data = other.genome[(3 * other_A) + 2:]

            A_crossover = max(self_A,other_A) - abs(self_A - other_A)
            B_crossover = max(self_B,other_B) - abs(self_B - other_B)

            new_A = self_A if random.random() < 0.5 else other_A
            new_B = self_B if random.random() < 0.5 else other_B

            new_A_data = [0 for i in range(new_A*3)]
            new_B_data = [0 for i in range(new_B*2)]

            tries = 0
            while not CNNGenome.validate_convolutional_genome_information(new_A_data, self.network_input_dimensionality):
                tries += 1
                for i in range(new_A * 3):
                    if i < A_crossover * 3:
                            new_A_data[i] = self_A_data[i] if random.random() < 0.5 else other_A_data[i]
                    else:
                            new_A_data[i] = self_A_data[i] if self_A > other_A else other_A_data[i]
                if tries > 250:
                    print("TRIES TIMEOUT")
                    break
            if tries > 250:
                continue

            for i in range(new_B * 2):
                # print("i", i)
                if i < B_crossover * 2:
                        new_B_data[i] = self_B_data[i] if random.random() < 0.5 else other_B_data[i]
                else:
                        new_B_data[i] = self_B_data[i] if self_B > other_B else other_B_data[i]
            g = CNNGenome(self.network_input_dimensionality)
            g.set_genome([new_A, new_B] + new_A_data + new_B_data)
            return g

    def randomize(self, max_conv_layers, max_dense_layers):
        """ Sets this genome to be random
        """
        success = False
        tries = 0
        A = random.randint(1,max_conv_layers)
        B = random.randint(1,max_dense_layers)
        # print(max_conv_layers, max_dense_layers)
        a_data = [0 for i in range(A*3)]
        layer_no = 1

        while layer_no <= A:
            try:
                a_data[((layer_no-1)*3):((layer_no-1)*3) + 3] = CNNGenome.generate_random_conv_layer_attributes(layer_no, a_data, self.network_input_dimensionality)
                layer_no += 1
            except ValueError as b:
                layer_no -= 1
        b_data = []
        for i in range(B):
            b_data += CNNGenome.generate_random_dense_layer_attributes()
        self.set_genome([A] + [B] + a_data + b_data)

    # TODO max size representation: use window size variable instead of hardcoded value
    def set_to_max_size(self):
        """ sets this genome to represent the max size nn architecture it possibly can
        """
        self.set_genome([2,3,2,1,1024,2,1,1024,1000,0,1000,0,1000,0])

    # TODO min size representation: use window size variable instead of hardcoded value
    def set_to_min_size(self):
        """ sets this genome to represent the min size nn architecture it possibly can
        """
        self.set_genome([1,1,9,1,4,1,0])

    @staticmethod
    def create_random(max_cnn_layers=5, max_dense_layers=5):
        """ Creates a new randomized genome """
        g = CNNGenome()
        g.randomize(max_cnn_layers, max_dense_layers)
        return g

    def get_conv_layer_count(self):
        return 0 if len(self.genome) < 7 else self.genome[0]

    def get_dense_layer_count(self):
        return 0 if len(self.genome) < 7 else self.genome[1]

    def get_conv_layer_attributes(self, layer_no):
        """ Layer Number starts with 1!
            Returns a 3-tuple (filters,kernel_size,strides)
        """
        if layer_no < 1:
            raise ValueError("Layers are numbered beginning from 1.")
        elif len(self.genome) < 7 or layer_no > self.genome[0]:
            return None
        i = 2 + ((layer_no-1)*3)
        return [int(x) for x in tuple(self.genome[i:i+3])]

    def get_dense_layer_attributes(self, layer_no):
        """ Layer Number starts with 1!
        Returns a 2-tuple: (integer layer_size, float dropout_rate)
        """
        if layer_no < 1:
            raise ValueError("Layers are numbered beginning from 1.")
        elif len(self.genome) < 7 or layer_no > self.genome[1]:
            return None
        i = 2 + (self.genome[0]*3) + (2 *(layer_no-1))
        return [x for x in tuple(self.genome[i:i+2])]

    @staticmethod
    def validate_genome(genome):
        if type(genome) != type(list()):
            return False
        elif len(genome) < 7:
            return False
        else:
            A = genome[0]
            B = genome[1]
            if len(genome) != 2 + (A * 3) + (B * 2):
                return False
            else:
                for i in range(2,len(genome[:(A*3)+2])):
                    if genome[i] < 1:
                        return False
                offset = len(genome[:(A*3)+2])
                for i in range(0,len(genome) - offset):
                    if i % 2 == 0:
                        if genome[i + offset] < 1:
                            return False
                    else:
                        if genome[i + offset] > 19 or genome[i] < 0:
                            return False
        return True

    @staticmethod
    def random_neuron_count():
        return random.randint(1,1000)

    @staticmethod
    def random_dropout_factor():
        return random.randint(0,16)

    @staticmethod
    def random_filter_count():
        return math.pow(2,random.randint(2,10))

    @staticmethod
    def validate_convolutional_genome_information(cnn_layers_information, input_length):
        """ Takes a list of [kernel_size, strides, filters, kernel_size, strides, ...]
            Returns False if this architecture is impossible, True otherwise.
        """
        layer_count = int(len(cnn_layers_information) / 3)
        input_dim = input_length
        try:
            for i in range(0, layer_count):
                kernel_length = cnn_layers_information[(i*3)]
                if kernel_length > input_dim:
                    return False
                input_dim = CNNGenome.get_cnn_layer_output_dimensionality(cnn_layers_information, input_length, i+1)
            return True
        except ValueError:
            return False

    @staticmethod
    def generate_random_dense_layer_attributes():
        """ returns genome information for a random dense layer.
            returns a tuple (nodes, dropout_factor)
        """
        return(CNNGenome.random_neuron_count(), CNNGenome.random_dropout_factor())

    @staticmethod
    def generate_random_conv_layer_attributes(layer_depth, existing_conv_layers, input_dim):
        """ existing_layers: [kernel_size_1, strides_1, filters_1, ..., filters_n]
            layer_depth: Output will be generated for a cnn layer of depth layer_depth with
            respect to the provided existing_layers information.
        """
        if layer_depth > 1:
            current_input_dim = CNNGenome.get_cnn_layer_output_dimensionality(existing_conv_layers, input_dim, layer_depth-1)
        else:
            current_input_dim = input_dim
        if current_input_dim < 2:
            raise ValueError("Tried to generate CNN layer information for input length 1 (or less)")
        kernel = random.randint(2,current_input_dim) if current_input_dim >= 2 else 1
        strides = random.randint(1,max(1,min(kernel, current_input_dim-kernel)))
        filters = CNNGenome.random_filter_count()
        return [kernel, strides, filters]

    @staticmethod
    def get_cnn_layer_output_dimensionality(cnn_layers_information, input_length, n):
        """ layers: [kernel_size_1, strides_1, filters_1, ..., filters_n]
            Returns output dim of layer n
        """
        layer_out_dim = input_length
        for index in range(0,n*3,3):
            kernel_size = cnn_layers_information[index]
            strides = cnn_layers_information[index+1]
            if strides < 1:
                raise ValueError("Strides smaller than 1")
            layer_out_dim = math.ceil((1 + (layer_out_dim - kernel_size)) / strides)
            layer_out_dim = math.ceil(layer_out_dim / 2.0)
            # print(layer_out_dim)
            if kernel_size > input_length or layer_out_dim < 1:
                raise ValueError("Convolution swallows input!")
        return layer_out_dim

    def clone(self):
        g = CNNGenome(self.network_input_dimensionality)
        g.set_genome(self.genome[:])
        return g

    def __iter__(self):
        return CNNGenomeIterator(self)
    def __str__(self):
        return "CNNGenome[conv:3*" + str(self.genome[0]) + "|dense:1*" + str(self.genome[1]) + "] " + str(self.genome[2:])
    def __repr__(self):
        return "CNNGenome[conv:3*" + str(self.genome[0]) + "|dense:1*" + str(self.genome[1]) + "] " + str(self.genome[2:])
    def __bool__(self):
        return len(self.genome) > 0

class CNNGenomeIterator():
    """ Iterator class for iteration along a CNNGenome """
    def __init__(self, cnn_genome):
        self.container = cnn_genome
        self.index = 0
    def __next__(self):
        if self.index >= len(self.container.genome):
            raise StopIteration
        elem = self.container.genome[self.index]
        self.index += 1
        return elem

def weighted_choice(choices):
    """ choices must be provided as a list of (object,weighting) tuples
    """
    max = sum((choice[1] for choice in choices))
    pick = random.uniform(0, max)
    current = 0
    for obj, weight in choices:
        current += weight
        if current > pick:
            return obj

def raise_(exc):
    raise exc

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
