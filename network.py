"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
## Se importan librerias random y numpy
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):
#Se crea la clase Network

    def __init__(self, sizes):
    #Se hace el __init__ para inicializar la clase
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        ## Se establece el numero de capas en la red y el número de neuronas en cada capa
        self.num_layers = len(sizes)
        self.sizes = sizes
        ## Se inicia con self. los sesgos (vectores de cada capa menos de la de entrada) y pesos (que conectan las capas) de manera aleatoria
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        #Se define la función feedforward
        """Return the output of the network if ``a`` is input."""
        ## Se itera a traves de las capas de la red
        ##  Se calcula la salida de la capa actual utilizando la función de activación sigmoide y 
        # np.dot realiza el producto punto entre los pesos y la entrada, luego se suma el sesgo
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        #Devuelve a
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
                # Se crea la función para el Stochastic Gradient Descen
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        ## Para complementar la parte de arriba, epochs es el número de veces que iteramos sobre todos los datos de entrenamiento
        ## y eta es la taza de aprendizaje
        if test_data:
        #Si la variable test_data es diferente de cero
            test_data = list(test_data)
            #La variable se hace lista
            n_test = len(test_data)
            #Se crea la variable n_test a la cantidad de elementos de la lista

        training_data = list(training_data)
        #Vuelve a training_data en una lista
        n = len(training_data)
        #Asigna a n el valor de la cantidad de elementos en la lista
        for j in range(epochs):
        #Por cada elemento en el rango de epochs
            ## Se barajean los datos de entrenamiento para evitar sesgos en el aprendizaje
            random.shuffle(training_data)
            ## Se definen los minibatches al dividir los datos en batches de tamaño _mini_batch_size
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
            #Se hace un ciclo for para todos los elementos de mini_batches
                ## Se actualizan los pesos y sesgos utilizando el mini-batch actual
                self.update_mini_batch(mini_batch, eta)
            if test_data:
            #Si test_data es diferente a cero se imprime un mensaje con la epoca actual, rendimiento de la red y numero de datos de prueba
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
                #Si no, envia el mensaje de que se ha terminado una epoca
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
    #Se crea la función u_m_b
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``"""
        #Se inician las listas nabla_b y _w con matrices de ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
        #Ciclo for para cada (x, y) en el mini batch actual
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #Se hace la retropropagación 
            ## Acumulamos los  gradientes para cada ejemplo en el mini-batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # Se actualizan los pesos y sesgos utilizando los gradientes acumulados, 
            # Combina los pesos actuales con los gradientes de la retropropagación
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #Se inician las listas nabla_b y _w con matrices de ceros.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Se inicia la variable activation y la lista activations cuyo primer elemento sera x
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        ## Esta es la propagación hacia adelante: se calculan activaciones y z para cada capa con la función sigmoide
        for b, w in zip(self.biases, self.weights):
        #Bucle para cada peso y sesgo
            z = np.dot(w, activation)+b # Se calcula z, wx + b
            zs.append(z) #Se agrega el valor de z a la lista zs
            activation = sigmoid(z)
            #Se define activation como la sigmoide evaluada en z
            activations.append(activation)
            #Se agrega el valor a la lista activations
        # backward pass: Calculamos los gradientes de la función de costo
        delta = self.cost_derivative(activations[-1], y) * \
        #Se calcula el error en la capa de salida
            sigmoid_prime(zs[-1])
        #Se asigna el error calculado en la capa de salida a la última entrada de la lista nabla_b
        nabla_b[-1] = delta
        #Se calcula el gradiente de los pesos para la capa de salida.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
        #Bucle para las capas ocultas de la red
            z = zs[-l] # z representa la entrada ponderada a la capa oculta actual.
            sp = sigmoid_prime(z) #sp guarda la derivada de la sigmoide evaluada en z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #Con retropropagación se calcula el error de la capa oculta acutal
            nabla_b[-l] = delta #Se asigna la salida a una lista
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #Se calcula el gradiente de los pesos
        return (nabla_b, nabla_w) #Devuelve gradientes de sesgos y pesos

    def evaluate(self, test_data):
    #Se crea la función evaluate
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        ## Repitiendo lo que se explica arriba, se comparan las salidas predichas con las salidas reales y se cuentan las correctas
        test_results = [(np.argmax(self.feedforward(x)), y) #Se evalua la red neuronal en cada ejemplo del conjunto de datos de prueba
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) 
        #Calcula el número total de predicciones correctas en el conjunto de datos de prueba.

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        ## La derivada de la función de costo respecto a la salida de la red
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    ## Se aplica la función sigmoide a Z
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    ## Se aplica la derivada de la función sigmoide para calcular cómo los pesos de la red deben ajustarse para minimizar el error
    return sigmoid(z)*(1-sigmoid(z))
