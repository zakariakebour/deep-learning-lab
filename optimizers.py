class Optimizer:
    def __init__(self, lr):
        self.lr = lr  # tasa de aprendizaje (antes era "step", ahora es learning rate)

    def update(self, neurona, x, real):
        # Un solo forward — ya no repetimos esto por cada peso
        pred = neurona.forward(x)

        # dL/dpred: cuánto cambia el loss si cambia la predicción
        d_loss_d_pred = -2 * (real - pred)

        # Para cada peso: dL/dw_i = dL/dpred * dpred/dw_i = d_loss_d_pred * x[i]
        for i in range(len(neurona.weights)):
            gradiente = d_loss_d_pred * x[i]

            neurona.weights[i] -= self.lr * gradiente

            print(f"Peso {i+1}: Gradiente: {gradiente}")

        # dL/dbias = dL/dpred * dpred/dbias = d_loss_d_pred * 1
        d_loss_d_bias = d_loss_d_pred
        neurona.bias -= self.lr * d_loss_d_bias

          