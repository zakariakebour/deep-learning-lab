from losses import MSE

class Trainer:

    def train(self, network, dataset,optimizador):

        epoch = 10

        for i in range(epoch):
            for x, real in dataset:

                # Hacer la predicción
                pred = network.forward(x)

                # Convertimos el numero de salida de la ultima capa en un valor normal y no en una lista
                pred = pred[0]

                # Calcular la pérdida
                loss = MSE.compute(real, pred)

                # Mostrar información
                print(
                    f"Epoch: {i + 1}\n"
                    f"x={x} | "
                    f"pred={pred} | "
                    f"real={real} | "
                    f"loss={loss} | "
                )

                #Aqui el optimizador
                optimizador.update(network,x,real)
