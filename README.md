# Mini IA — Red neuronal construida desde cero

Implementación educativa de una red neuronal *fully connected* (densa) en Python puro, sin frameworks como PyTorch o TensorFlow. El objetivo del proyecto es entender e implementar manualmente el mecanismo interno del *forward pass*, la función de pérdida y el *backpropagation* mediante la regla de la cadena.

---

## Tabla de contenidos

- [Objetivo del proyecto](#objetivo-del-proyecto)
- [Arquitectura](#arquitectura)
- [Estructura de archivos](#estructura-de-archivos)
- [Conceptos matemáticos aplicados](#conceptos-matemáticos-aplicados)
- [Flujo de ejecución](#flujo-de-ejecución)
- [Resultados observados](#resultados-observados)
- [Bugs encontrados y corregidos](#bugs-encontrados-y-corregidos)
- [Limitaciones actuales](#limitaciones-actuales)
- [Próximos pasos](#próximos-pasos)

---

## Objetivo del proyecto

Construir, sin librerías externas de deep learning, los componentes mínimos necesarios para que una red neuronal aprenda:

1. **Neurona**: unidad básica que combina entradas, pesos y bias.
2. **Capa (`Layer`)**: conjunto de neuronas en paralelo que comparten la misma entrada.
3. **Red (`Network`)**: secuencia de capas encadenadas (la salida de una alimenta a la siguiente).
4. **Optimizador**: cálculo del gradiente vía *backpropagation* y actualización de pesos.
5. **Entrenador (`Trainer`)**: bucle de entrenamiento sobre un dataset a lo largo de varias épocas.

---

## Arquitectura

```
Entrada (x)
    │
    ▼
┌─────────────┐
│   Layer 1   │   2 neuronas, cada una con 2 pesos + bias, activación ReLU
└─────────────┘
    │  vector de 2 valores
    ▼
┌─────────────┐
│   Layer 2   │   1 neurona (capa de salida), 2 pesos + bias, activación ReLU
└─────────────┘
    │
    ▼
Predicción (1 valor) ──► comparar con "real" ──► Loss (MSE)
```

**Regla de diseño clave:** el número de neuronas de la última capa debe coincidir con la forma del valor que se quiere predecir. Como el dataset predice un único número (por ejemplo, un precio), la capa de salida tiene **una sola neurona**.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `neurona.py` | Clase `Neuron`: realiza `z = w·x + b`, aplica ReLU, guarda `z` y `last_input` para el backward |
| `layer.py` | Clase `Layer`: ejecuta todas sus neuronas con la misma entrada y devuelve un vector de salidas |
| `network.py` | Clase `Network`: encadena capas, pasando la salida de una como entrada de la siguiente |
| `activations.py` | Función de activación `ReLU` |
| `losses.py` | Función de pérdida `MSE` (Mean Squared Error) |
| `optimizers.py` | Clase `Optimizer`: calcula gradientes vía regla de la cadena (backpropagation) y actualiza pesos y bias |
| `trainer.py` | Clase `Trainer`: bucle de entrenamiento sobre épocas y ejemplos del dataset |
| `datasets.py` | Dataset de ejemplo: pares `(x, real)` |
| `main.py` | Punto de entrada: construye la red, entrena y ejecuta una predicción final |

---

## Conceptos matemáticos aplicados

### 1. Forward pass

Cada neurona calcula:

$$z = \sum_i w_i x_i + b \qquad a = \text{ReLU}(z)$$

### 2. Función de pérdida (MSE)

$$L = (y_{pred} - y_{real})^2$$

### 3. Regla de la cadena (backpropagation)

La idea central del proyecto: para saber cómo un peso lejano de la salida afecta al error, se recorren los tramos intermedios y se multiplican sus derivadas.

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Con las fórmulas concretas usadas en el proyecto:

$$\frac{\partial L}{\partial y_{pred}} = 2(y_{pred} - y_{real}) \qquad
\frac{\partial a}{\partial z} = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases} \qquad
\frac{\partial z}{\partial w_i} = x_i$$

### 4. Propagación del error entre capas (deltas)

Para una neurona que **no** está en la capa de salida, su "delta" (sensibilidad al error) depende de los deltas de **todas** las neuronas de la capa siguiente a las que está conectada:

$$\delta_j^{(l)} = \left( \sum_{k} \delta_k^{(l+1)} \cdot w_{k,j}^{(l+1)} \right) \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}}$$

Esto es lo que permite que el error, calculado solo al final de la red, se reparta hacia atrás por todas las capas anteriores — de ahí el nombre *backpropagation*.

### 5. Actualización de pesos (descenso de gradiente)

$$w_{nuevo} = w - \eta \cdot \frac{\partial L}{\partial w}$$

donde $\eta$ (`lr`) es la tasa de aprendizaje.

---

## Flujo de ejecución

1. `main.py` crea las neuronas de cada capa con pesos y bias iniciales.
2. `Trainer.train()` recorre el dataset durante `N` épocas.
3. Para cada ejemplo `(x, real)`:
   - `Network.forward(x)` propaga la entrada capa por capa.
   - Se calcula el `loss` con `MSE`.
   - `Optimizer.update()` ejecuta el backward: calcula los deltas desde la última capa hacia la primera y actualiza pesos y bias de cada neurona.
4. Al finalizar el entrenamiento, se ejecuta una predicción final con `network.forward(...)`.

**Detalle importante de la implementación:** el entrenamiento se realiza en modalidad **SGD puro** (Stochastic Gradient Descent) — los pesos se actualizan después de **cada ejemplo individual**, no después de promediar varios. Esto significa que los mismos pesos son "negociados" secuencialmente por todos los ejemplos del dataset en cada época.

---

## Resultados observados

Con el dataset de 4 ejemplos y `lr = 0.001` durante 100 épocas:

| Ejemplo (`x`) | Valor real | Loss inicial | Loss final (época 100) |
|---|---|---|---|
| `[2, 1]` | 7 | ~0.77 | ~2.05 (estabilizado) |
| `[3, 2]` | 11 | ~3.23 | ~2.54 |
| `[4, 3]` | 15 | ~7.07 | ~0.61 |
| `[3.5, 2]` | 8 | ~36.0 | ~11.47 (no converge) |

**Observación clave:** el ejemplo `[3.5, 2] → 8` no logra reducir su error de forma consistente, mientras que los otros tres sí. Esto se debe a que, con solo 4 puntos de datos, ese ejemplo **no es compatible con el mismo patrón lineal** que explica a los otros tres — los pesos compartidos llegan a un punto de compromiso, no a una solución perfecta para los cuatro.

**Conclusión validada:** el mecanismo de forward + loss + backpropagation multicapa funciona correctamente (los gradientes se calculan y aplican según la regla de la cadena). La falta de convergencia total es una limitación de los datos, no del algoritmo.

---

## Bugs encontrados y corregidos

Durante el desarrollo se identificaron y corrigieron los siguientes errores, documentados aquí por su valor didáctico:

1. **`Layer.forward` sin `return`**: el método calculaba el vector de salida pero no lo devolvía, provocando que la capa siempre entregara `None`.
2. **`Network.forward` no encadenaba las capas**: se llamaba a `layer.forward(x)` con la entrada original en cada iteración, en lugar de `layer.forward(entrada)` con la salida acumulada de la capa anterior.
3. **Desajuste de dimensiones silencioso (`zip`)**: al pasar una entrada de 3 valores a neuronas con solo 2 pesos, `zip()` truncó el vector sin lanzar ningún error, produciendo un resultado numéricamente "válido" pero incorrecto.
4. **Última capa con más de una neurona**: al tener 2 neuronas en la capa de salida, la red devolvía un vector `[a, b]` que no podía compararse contra un `real` escalar. Se corrigió reduciendo la capa de salida a 1 neurona.
5. **Variable de bucle reutilizando un nombre reservado**: `for x, self.weights in zip(self.weights, x)` sobrescribía `self.weights` (la lista de pesos del objeto) con un valor numérico individual en cada iteración, corrompiendo el estado de la neurona para llamadas posteriores.
6. **Falta de `last_input` en la neurona**: el backward necesita conocer la entrada que recibió cada neurona en su último `forward` para calcular $\partial z / \partial w_i = x_i$; no se guardaba inicialmente.
7. **`Optimizer.update` diseñado para una sola neurona**: la primera versión no contemplaba múltiples capas ni múltiples neuronas por capa; fue reescrita para calcular y propagar deltas por capa, de atrás hacia adelante.
8. **Inconsistencia de nombres de atributo** (`nueronas` vs `neuronas`): error de tipeo que provocó un `AttributeError` al no coincidir con el atributo real de la clase `Layer`.

---

## Limitaciones actuales

Este proyecto implementa el **núcleo algorítmico** de una red neuronal, pero carece de componentes necesarios para un caso de uso real (por ejemplo, predicción de precios de vivienda):

- **Dataset mínimo** (4 ejemplos) — insuficiente para generalizar patrones reales.
- **Sin separación entrenamiento/prueba (train/test split)** — no hay forma de validar si la red generaliza a datos no vistos.
- **Sin normalización de entradas** — variables con escalas muy distintas afectarían negativamente el entrenamiento.
- **Entrenamiento SGD puro únicamente** — sin soporte para *mini-batch*, que es el estándar en la práctica profesional.
- **Sin regularización** (weight decay, dropout) — riesgo de sobreajuste con datasets más grandes.
- **Optimizador básico** — sin momentum ni tasas de aprendizaje adaptativas (Adam, RMSprop).
- **Riesgo de neuronas "muertas"** — con ReLU y pocas neuronas por capa, una neurona que cae permanentemente en zona negativa deja de aprender.
