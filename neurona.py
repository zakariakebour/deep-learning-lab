# Importamos el método de función de activación ReLU
from activations import Activation

# Clase neurona
class Neuron:
    def __init__(self,weights,bias,z):
        # Guardamos la ultima entrada de cada neurona para poder aplicar despues el backpropagation a cada neurona
        self.last_input = None
        self.weights = weights
        # Guardamos bias
        self.bias = bias
        # Guardamos la operación z
        self.z = None

    # Metodo para prediccion
    def forward(self,x):
        # Variable para acumular el resultado de todas las multiplicaciones
        z = 0
        
        # Realizamos la operación: Cada peso se multiplica por su entrada y se añade el bias al resultado total
        self.last_input = x  # guardamos la entrada para el backward
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

        # Guardamos el valor z 
        self.z = z
        
        # Aplicamos función de activación al resultado de la operación z
        activacion = Activation.ReLU(z)

        # Devolvemos la predicción
        return activacion

