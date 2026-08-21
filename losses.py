# Clase para calcular Loss (MSE)
class MSE:

    @staticmethod
    def compute(real, pred):
        return (real - pred) ** 2
        
