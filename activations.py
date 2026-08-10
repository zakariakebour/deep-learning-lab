class Activation:

    @staticmethod
    def ReLU(z):
        # Si el resultado de z es negativo lo dejamos 0
        if z < 0:
            return 0
        else:
            return z

    # Se podria realizar de forma más corta y profesional usando el método max()
    # def ReLU(z):
        # return max(0,z)