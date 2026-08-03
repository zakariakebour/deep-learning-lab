from losses import MSE

class Trainer:

    def train(self, neurona, dataset,optimizador):

        for x, real in dataset:

            # 1. Hacer la predicción
            pred = neurona.forward(x)

            # 2. Calcular la pérdida
            loss = MSE.compute(real, pred)

            # 3. Mostrar información
            print(
                f"x={x} | "
                f"pred={pred} | "
                f"real={real} | "
                f"loss={loss} | "
                f"peso={neurona.weights} "
            )

           #Aqui el optimizador
            optimizador.update(neurona,x,real)

