#Clase para calcular Loss
class MSE:

    @staticmethod
    def compute(real, pred):
        return (real - pred) ** 2
        
