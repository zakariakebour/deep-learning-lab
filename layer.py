# Clase layer (capa que contiene varias neuronas)
class Layer:
    def __init__(self,neuronas):
        # Parámetro para registrar las neuronas
        self.neuronas = neuronas

    # Método forward
    def forward(self,x):
        # Vector 
        vector_salida = []

        for neurona in self.neuronas:
            # Cada salida es una operacion de una neurona
            salida = neurona.forward(x)

            # Añadimos la salida al vector
            vector_salida.append(salida)

        # El método devuelve el vector completo con las salidas de las neuronas de toda la capa
        return vector_salida

     