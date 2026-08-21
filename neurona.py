# Importamos el método de función de activación ReLU
from activations import Activation

# Clase neurona
class Neuron:
    def __init__(self,weights,bias,z):
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
        z = sum(x * self.weights for x, self.weights in zip(self.weights,x)) + self.bias

        # Guardamos el valor z 
        self.z = z
        
        # Aplicamos función de activación al resultado de la operación z
        activacion = Activation.ReLU(z)

        # Devolvemos la predicción
        return activacion

