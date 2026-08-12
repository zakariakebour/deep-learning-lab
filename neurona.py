# Importamos el método de función de activación ReLU
from activations import Activation

# Clase neurona
class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        # Guardamos bias
        self.bias = bias

    # Metodo para prediccion
    def forward(self,x):
        # Variable para acumular el resultado de todas las multiplicaciones
        z = 0
        
        for peso, entrada in zip(self.weights,x):
            z += peso * entrada

        # Sumamos bias 
        z += self.bias

        # Aplicamos función de activación al resultado de la operación z
        activacion = Activation.ReLU(z)

        # Devolvemos la predicción
        return activacion

