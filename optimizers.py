class Optimizer:
    # Constructor
    def __init__(self, lr):
        self.lr = lr  # tasa de aprendizaje (antes era "step", ahora es learning rate)

    def update(self, network, x, real):
        # Un solo forward — ya no repetimos esto por cada peso
        # (cada neurona ya guardó su z y su last_input durante este forward)
        pred = network.forward(x)

        # Es necesario convertir el valor a numero sin estar dentro de una lista para poder restarlo al valor real
        pred = pred[0]

        # dL/dpred: cuánto cambia el loss si cambia la predicción
        d_loss_d_pred = -2 * (real - pred)

        # Delta de la última capa (normalmente 1 neurona en capa de salida)
        output_layer = network.layers[-1]
        deltas = []

        # Recorremos las neuronas de la última capa y calculamos el delta de cada una
        for neurona in output_layer.neuronas:
            # dReLU/dz: cuánto cambia ReLU si cambia z
            d_ReLU_d_z = 1 if neurona.z > 0 else 0

            # delta = dL/dpred * dpred/dz = d_loss_d_pred * d_ReLU_d_z
            deltas.append(d_loss_d_pred * d_ReLU_d_z)

        # Recorremos las capas de atrás hacia adelante (backpropagation)
        for l in reversed(range(len(network.layers))):
            layer = network.layers[l]

            # Para cada neurona de esta capa: actualizamos sus pesos y su bias con el delta actual
            for n_idx, neurona in enumerate(layer.neuronas):
                delta = deltas[n_idx]

                # Para cada peso: dL/dw_i = delta * dz/dw_i = delta * last_input[i]
                for w_idx in range(len(neurona.weights)):
                    gradiente = delta * neurona.last_input[w_idx]

                    neurona.weights[w_idx] -= self.lr * gradiente

                    print(f"Capa {l+1} - Neurona {n_idx+1} - Peso {w_idx+1}: Gradiente: {gradiente}")

                # dL/dbias = delta * dz/dbias = delta * 1
                neurona.bias -= self.lr * delta

            # Si hay una capa anterior, calculamos sus deltas antes de seguir hacia atrás
            if l > 0:
                prev_layer = network.layers[l - 1]
                nuevos_deltas = []

                # Recorremos cada neurona de la capa anterior
                for j, neurona_prev in enumerate(prev_layer.neuronas):
                    # Sumamos la influencia de todas las neuronas de la capa actual,
                    # cada una pesada por el peso que la conecta con esta neurona anterior
                    suma = 0
                    for k, neurona_actual in enumerate(layer.neuronas):
                        suma += deltas[k] * neurona_actual.weights[j]

                    # dReLU/dz de la neurona anterior
                    d_ReLU_d_z = 1 if neurona_prev.z > 0 else 0

                    nuevos_deltas.append(suma * d_ReLU_d_z)

                # Estos nuevos deltas serán los que use la siguiente vuelta del bucle (capa anterior)
                deltas = nuevos_deltas