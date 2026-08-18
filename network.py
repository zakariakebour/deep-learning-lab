# Creamos la clase network que guardara los layers para crear una red completa en orden
class Network:
    # Constructor
    def __init__(self,layers):
        self.layers = layers

    # Método para pasar el resultado del layer a su siguiente capa
    def forward(self,x,layer):
        # Entrada
        entrada = x

        # Recorremos la red que contiene las capas o el resultado de cada layer
        for layer in self.layers:
            # Ejecutamos el primer layer con la entrada x
            salida = layer.forward(entrada)

            # Ahora la salida del primer layer es la entrada
            entrada = salida

        # Devolvemos la entrada
        return entrada