from losses import MSE

class Trainer:

    def train(self, neurona, dataset,optimizador):

        epoch = 10

        for i in range(epoch):
            for x, real in dataset:

                #Hacer la predicción
                pred = neurona.forward(x)

                #Calcular la pérdida
                loss = MSE.compute(real, pred)

                #Mostrar información
                print(
                    f"Epoch: {i + 1}\n"
                    f"x={x} | "
                    f"pred={pred} | "
                    f"real={real} | "
                    f"loss={loss} | "
                    f"peso={neurona.weights} "
                )

                #Aqui el optimizador
                optimizador.update(neurona,x,real)

