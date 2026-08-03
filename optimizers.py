from losses import MSE

class Optimizer:
    def __init__(self,step):
        self.step = step

    def update(self,neurona,x,real):
        for i in range(len(neurona.weights)):
            #Guardamos el peso original
            peso_original = neurona.weights[i]

            #Calculamos el loss actual para saber como esta actualmente el modelo
            pred_actual = neurona.forward(x)

            loss_actual = MSE.compute(real,pred_actual)

            #Al peso le añadimos el valor nuevo
            neurona.weights[i] += self.step
   
            #Prediccion nueva con el cambio de peso
            pred_nueva = neurona.forward(x)

            loss_nueva = MSE.compute(real,pred_nueva)

            #Aproximar el gradiante
            gradiante = (loss_nueva - loss_actual) / self.step

            #Imprimimos el gradiante
            print(f"Peso {i + 1}: Gradiante: {gradiante}")
              
            #Aplicamos la mejora a los pesos
            neurona.weights[i] = peso_original - self.step * gradiante


