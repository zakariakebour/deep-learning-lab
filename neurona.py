#Clase neurona
class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        #Guardamos bias
        self.bias = bias

    #Metodo para prediccion
    def forward(self,x):
        #Variable para acumular el resultado de todas las multiplicaciones
        z = 0
        
        for peso, entrada in zip(self.weights,x):
            z += peso * entrada

        #Sumamos bias 
        z += self.bias

        #Devolvemos el resultado
        return z

