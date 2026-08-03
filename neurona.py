#Clase neurona
class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        #Guardamos bias
        self.bias = bias

    #Metodo para prediccion
    def forward(self,x):
        #Variable para acumular el resultado de todas las multiplicaciones
        resultado = 0
        
        for peso, entrada in zip(self.weights,x):
            resultado += peso * entrada

        #Sumamos bias 
        resultado += self.bias

        #Devolvemos el resultado
        return resultado

