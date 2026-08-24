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

Construir, sin librerías externas de *deep learning*, los componentes mínimos necesarios para que una red neuronal aprenda:

* **Neurona:** Unidad básica que combina entradas, pesos y *bias*.
* **Capa (Layer):** Conjunto de neuronas en paralelo que comparten la misma entrada.
* **Red (Network):** Secuencia de capas encadenadas (la salida de una alimenta a la siguiente).
* **Optimizador:** Cálculo del gradiente vía *backpropagation* y actualización de pesos.
* **Entrenador (Trainer):** Bucle de entrenamiento sobre un dataset a lo largo de varias épocas.

---

## Arquitectura

```text
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
Regla de diseño clave: El número de neuronas de la última capa debe coincidir con la forma del valor que se quiere predecir. Como el dataset predice un único número (por ejemplo, un precio), la capa de salida tiene una sola neurona.

Estructura de archivos
Archivo	Responsabilidad
neurona.py	Clase Neuron: realiza z=w⋅x+b, aplica ReLU, guarda z y last_input para el backward.
layer.py	Clase Layer: ejecuta todas sus neuronas con la misma entrada y devuelve un vector de salidas.
network.py	Clase Network: encadena capas, pasando la salida de una como entrada de la siguiente.
activations.py	Función de activación ReLU.
losses.py	Función de pérdida MSE (Mean Squared Error).
optimizers.py	Clase Optimizer: calcula gradientes vía regla de la cadena (backpropagation) y actualiza pesos y bias.
trainer.py	Clase Trainer: bucle de entrenamiento sobre épocas y ejemplos del dataset.
datasets.py	Dataset de ejemplo: pares (x, real).
main.py	Punto de entrada: construye la red, entrena y ejecuta una predicción final.
Conceptos matemáticos aplicados
1. Forward pass
Cada neurona calcula:

z= 
i
∑
​
 w 
i
​
 x 
i
​
 +ba=ReLU(z)
2. Función de pérdida (MSE)
L=(y 
pred
​
 −y 
real
​
 ) 
2
 
3. Regla de la cadena (backpropagation)
La idea central del proyecto: para saber cómo un peso lejano de la salida afecta al error, se recorren los tramos intermedios y se multiplican sus derivadas.

∂w
∂L
​
 = 
∂a
∂L
​
 ⋅ 
∂z
∂a
​
 ⋅ 
∂w
∂z
​
 
Con las fórmulas concretas usadas en el proyecto:

∂y 
pred
​
 
∂L
​
 =2(y 
pred
​
 −y 
real
​
 ) 
∂z
∂a
​
 ={ 
1
0
​
  
z>0
z≤0
​
  
∂w 
i
​
 
∂z
​
 =x 
i
​
 
4. Propagación del error entre capas (deltas)
Para una neurona que no está en la capa de salida, su "delta" (sensibilidad al error) depende de los deltas de todas las neuronas de la capa siguiente a las que está conectada:

δ 
j
(l)
​
 =( 
k
∑
​
 δ 
k
(l+1)
​
 ⋅w 
k,j
(l+1)
​
 )⋅ 
∂z 
j
(l)
​
 
∂a 
j
(l)
​
 
​
 
Esto es lo que permite que el error, calculado solo al final de la red, se reparta hacia atrás por todas las capas anteriores — de ahí el nombre backpropagation.

5. Actualización de pesos (descenso de gradiente)
w 
nuevo
​
 =w−η⋅ 
∂w
∂L
​
 
donde η (lr) es la tasa de aprendizaje.

Flujo de ejecución
main.py crea las neuronas de cada capa con pesos y bias iniciales.

Trainer.train() recorre el dataset durante N épocas.

Para cada ejemplo (x, real):

Network.forward(x) propaga la entrada capa por capa.

Se calcula el loss con MSE.

Optimizer.update() ejecuta el backward: calcula los deltas desde la última capa hacia la primera y actualiza pesos y bias de cada neurona.

Al finalizar el entrenamiento, se ejecuta una predicción final con network.forward(...).

Detalle importante de la implementación: El entrenamiento se realiza en modalidad SGD puro (Stochastic Gradient Descent) — los pesos se actualizan después de cada ejemplo individual, no después de promediar varios. Esto significa que los mismos pesos son "negociados" secuencialmente por todos los ejemplos del dataset en cada época.

Resultados observados
Con el dataset de 4 ejemplos y lr=0.001 durante 100 épocas:

Ejemplo (x)	Valor real	Loss inicial	Loss final (época 100)
[2, 1]	7	~0.77	~2.05 (estabilizado)
[3, 2]	11	~3.23	~2.54
[4, 3]	15	~7.07	~0.61
[3.5, 2]	8	~36.0	~11.47 (no converge)
Observación clave: El ejemplo [3.5, 2] → 8 no logra reducir su error de forma consistente, mientras que los otros tres sí. Esto se debe a que, con solo 4 puntos de datos, ese ejemplo no es compatible con el mismo patrón lineal que explica a los otros tres — los pesos compartidos llegan a un punto de compromiso, no a una solución perfecta para los cuatro.

Conclusión validada: El mecanismo de forward + loss + backpropagation multicapa funciona correctamente (los gradientes se calculan y aplican según la regla de la cadena). La falta de convergencia total es una limitación de los datos, no del algoritmo.

Bugs encontrados y corregidos
Durante el desarrollo se identificaron y corrigieron los siguientes errores, documentados aquí por su valor didáctico:

Layer.forward sin return: El método calculaba el vector de salida pero no lo devolvía, provocando que la capa siempre entregara None.

Network.forward no encadenaba las capas: Se llamaba a layer.forward(x) con la entrada original en cada iteración, en lugar de layer.forward(entrada) con la salida acumulada de la capa anterior.

Desajuste de dimensiones silencioso (zip): Al pasar una entrada de 3 valores a neuronas con solo 2 pesos, zip() truncó el vector sin lanzar ningún error, produciendo un resultado numéricamente "válido" pero incorrecto.

Última capa con más de una neurona: Al tener 2 neuronas en la capa de salida, la red devolvía un vector [a, b] que no podía compararse contra un real escalar. Se corrigió reduciendo la capa de salida a 1 neurona.

Variable de bucle reutilizando un nombre reservado: for x, self.weights in zip(self.weights, x) sobrescribía self.weights (la lista de pesos del objeto) con un valor numérico individual en cada iteración, corrompiendo el estado de la neurona para llamadas posteriores.

Falta de last_input en la neurona: El backward necesita conocer la entrada que recibió cada neurona en su último forward para calcular  
∂w 
i
​
 
∂z
​
 =x 
i
​
 ; no se guardaba inicialmente.

Optimizer.update diseñado para una sola neurona: La primera versión no contemplaba múltiples capas ni múltiples neuronas por capa; fue reescrita para calcular y propagar deltas por capa, de atrás hacia adelante.

Inconsistencia de nombres de atributo (nueronas vs neuronas): Error de tipeo que provocó un AttributeError al no coincidir con el atributo real de la clase Layer.

Limitaciones actuales
Este proyecto implementa el núcleo algorítmico de una red neuronal, pero carece de componentes necesarios para un caso de uso real (por ejemplo, predicción de precios de vivienda):

Dataset mínimo (4 ejemplos): Insuficiente para generalizar patrones reales.

Sin separación entrenamiento/prueba (train/test split): No hay forma de validar si la red generaliza a datos no vistos.

Sin normalización de entradas: Variables con escalas muy distintas afectarían negativamente el entrenamiento.

Entrenamiento SGD puro únicamente: Sin soporte para mini-batch, que es el estándar en la práctica profesional.

Sin regularización (weight decay, dropout): Riesgo de sobreajuste con datasets más grandes.

Optimizador básico: Sin momentum ni tasas de aprendizaje adaptativas (Adam, RMSprop).

Riesgo de neuronas "muertas": Con ReLU y pocas neuronas por capa, una neurona que cae permanentemente en zona negativa deja de aprender.

Próximos pasos
[ ] Añadir soporte para entrenamiento por mini-batches.

[ ] Implementar un módulo de división de datos (train/test split).

[ ] Agregar técnicas de normalización para las entradas.

[ ] Desarrollar e integrar el optimizador Adam.

pero damelo completo formato markdown no solo esta parte:

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



Construir, sin librerías externas de *deep learning*, los componentes mínimos necesarios para que una red neuronal aprenda:



* **Neurona:** Unidad básica que combina entradas, pesos y *bias*.

* **Capa (Layer):** Conjunto de neuronas en paralelo que comparten la misma entrada.

* **Red (Network):** Secuencia de capas encadenadas (la salida de una alimenta a la siguiente).

* **Optimizador:** Cálculo del gradiente vía *backpropagation* y actualización de pesos.

* **Entrenador (Trainer):** Bucle de entrenamiento sobre un dataset a lo largo de varias épocas.



---



## Arquitectura



```text

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

Regla de diseño clave: El número de neuronas de la última capa debe coincidir con la forma del valor que se quiere predecir. Como el dataset predice un único número (por ejemplo, un precio), la capa de salida tiene una sola neurona.

Estructura de archivos
Archivo	Responsabilidad
neurona.py	Clase Neuron: realiza z=w⋅x+b, aplica ReLU, guarda z y last_input para el backward.
layer.py	Clase Layer: ejecuta todas sus neuronas con la misma entrada y devuelve un vector de salidas.
network.py	Clase Network: encadena capas, pasando la salida de una como entrada de la siguiente.
activations.py	Función de activación ReLU.
losses.py	Función de pérdida MSE (Mean Squared Error).
optimizers.py	Clase Optimizer: calcula gradientes vía regla de la cadena (backpropagation) y actualiza pesos y bias.
trainer.py	Clase Trainer: bucle de entrenamiento sobre épocas y ejemplos del dataset.
datasets.py	Dataset de ejemplo: pares (x, real).
main.py	Punto de entrada: construye la red, entrena y ejecuta una predicción final.
Conceptos matemáticos aplicados
1. Forward pass
Cada neurona calcula:

z= 
i
∑
​
 w 
i
​
 x 
i
​
 +ba=ReLU(z)
2. Función de pérdida (MSE)
L=(y 
pred
​
 −y 
real
​
 ) 
2
 
3. Regla de la cadena (backpropagation)
La idea central del proyecto: para saber cómo un peso lejano de la salida afecta al error, se recorren los tramos intermedios y se multiplican sus derivadas.

∂w
∂L
​
 = 
∂a
∂L
​
 ⋅ 
∂z
∂a
​
 ⋅ 
∂w
∂z
​
 
Con las fórmulas concretas usadas en el proyecto:

∂y 
pred
​
 
∂L
​
 =2(y 
pred
​
 −y 
real
​
 ) 
∂z
∂a
​
 ={ 
1
0
​
  
z>0
z≤0
​
  
∂w 
i
​
 
∂z
​
 =x 
i
​
 
4. Propagación del error entre capas (deltas)
Para una neurona que no está en la capa de salida, su "delta" (sensibilidad al error) depende de los deltas de todas las neuronas de la capa siguiente a las que está conectada:

δ 
j
(l)
​
 =( 
k
∑
​
 δ 
k
(l+1)
​
 ⋅w 
k,j
(l+1)
​
 )⋅ 
∂z 
j
(l)
​
 
∂a 
j
(l)
​
 
​
 
Esto es lo que permite que el error, calculated solo al final de la red, se reparta hacia atrás por todas las capas anteriores — de ahí el nombre backpropagation.

5. Actualización de pesos (descenso de gradiente)
w 
nuevo
​
 =w−η⋅ 
∂w
∂L
​
 
donde η (lr) es la tasa de aprendizaje.

Flujo de ejecución
main.py crea las neuronas de cada capa con pesos y bias iniciales.

Trainer.train() recorre el dataset durante N épocas.

Para cada ejemplo (x, real):

Network.forward(x) propaga la entrada capa por capa.

Se calcula el loss con MSE.

Optimizer.update() ejecuta el backward: calcula los deltas desde la última capa hacia la primera y actualiza pesos y bias de cada neurona.

Al finalizar el entrenamiento, se ejecuta una predicción final con network.forward(...).

Detalle importante de la implementación: El entrenamiento se realiza en modalidad SGD puro (Stochastic Gradient Descent) — los pesos se actualizan después de cada ejemplo individual, no después de promediar varios. Esto significa que los mismos pesos son "negociados" secuencialmente por todos los ejemplos del dataset en cada época.

Resultados observados
Con el dataset de 4 ejemplos y lr=0.001 durante 100 épocas:

Ejemplo (x)	Valor real	Loss inicial	Loss final (época 100)
[2, 1]	7	~0.77	~2.05 (estabilizado)
[3, 2]	11	~3.23	~2.54
[4, 3]	15	~7.07	~0.61
[3.5, 2]	8	~36.0	~11.47 (no converge)
Observación clave: El ejemplo [3.5, 2] → 8 no logra reducir su error de forma consistente, mientras que los otros tres sí. Esto se debe a que, con solo 4 puntos de datos, ese ejemplo no es compatible con el mismo patrón lineal que explica a los otros tres — los pesos compartidos llegan a un punto de compromiso, no a una solución perfecta para los cuatro.

Conclusión validada: El mecanismo de forward + loss + backpropagation multicapa funciona correctamente (los gradientes se calculan y aplican según la regla de la cadena). La falta de convergencia total es una limitación de los datos, no del algoritmo.

Bugs encontrados y corregidos
Durante el desarrollo se identificaron y corrigieron los siguientes errores, documentados aquí por su valor didáctico:

Layer.forward sin return: El método calculaba el vector de salida pero no lo devolvía, provocando que la capa siempre entregara None.

Network.forward no encadenaba las capas: Se llamaba a layer.forward(x) con la entrada original en cada iteración, en lugar de layer.forward(entrada) con la salida acumulada de la capa anterior.

Desajuste de dimensiones silencioso (zip): Al pasar una entrada de 3 valores a neuronas con solo 2 pesos, zip() truncó el vector sin lanzar ningún error, produciendo un resultado numéricamente "válido" pero incorrecto.

Última capa con más de una neurona: Al tener 2 neuronas en la capa de salida, la red devolvía un vector [a, b] que no podía compararse contra un real escalar. Se corrigió reduciendo la capa de salida a 1 neurona.

Variable de bucle reutilizando un nombre reservado: for x, self.weights in zip(self.weights, x) sobrescribía self.weights (la lista de pesos del objeto) con un valor numérico individual en cada iteración, corrompiendo el estado de la neurona para llamadas posteriores.

Falta de last_input en la neurona: El backward necesita conocer la entrada que recibió cada neurona en su último forward para calcular  
∂w 
i
​
 
∂z
​
 =x 
i
​
 ; no se guardaba inicialmente.

Optimizer.update diseñado para una sola neurona: La primera versión no contemplaba múltiples capas ni múltiples neuronas por capa; fue reescrita para calcular y propagar deltas por capa, de atrás hacia adelante.

Inconsistencia de nombres de atributo (nueronas vs neuronas): Error de tipeo que provocó un AttributeError al no coincidir con el atributo real de la clase Layer.

Limitaciones actuales
Este proyecto implementa el núcleo algorítmico de una red neuronal, pero carece de componentes necesarios para un caso de uso real (por ejemplo, predicción de precios de vivienda):

Dataset mínimo (4 ejemplos): Insuficiente para generalizar patrones reales.

Sin separación entrenamiento/prueba (train/test split): No hay forma de validar si la red generaliza a datos no vistos.

Sin normalización de entradas: Variables con escalas muy distintas afectarían negativamente el entrenamiento.

Entrenamiento SGD puro únicamente: Sin soporte para mini-batch, que es el estándar en la práctica profesional.

Sin regularización (weight decay, dropout): Riesgo de sobreajuste con datasets más grandes.

Optimizador básico: Sin momentum ni tasas de aprendizaje adaptativas (Adam, RMSprop).

Riesgo de neuronas "muertas": Con ReLU y pocas neuronas por capa, una neurona que cae permanentemente en zona negativa deja de aprender.

Próximos pasos
[ ] Añadir soporte para entrenamiento por mini-batches.

[ ] Implementar un módulo de división de datos (train/test split).

[ ] Agregar técnicas de normalización para las entradas.

[ ] Desarrollar e integrar el optimizador Adam.
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(md_content)

print("FILE CREATED: README.md")


```text?code_stdout&code_event_index=1
FILE CREATED: README.md

Your Markdown file is ready

Icono de MD
README
 MD 
Abierto
Aquí tienes el documento completo listo para copiar y pegar directamente en tu archivo README.md:

Markdown
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

Construir, sin librerías externas de *deep learning*, los componentes mínimos necesarios para que una red neuronal aprenda:

* **Neurona:** Unidad básica que combina entradas, pesos y *bias*.
* **Capa (Layer):** Conjunto de neuronas en paralelo que comparten la misma entrada.
* **Red (Network):** Secuencia de capas encadenadas (la salida de una alimenta a la siguiente).
* **Optimizador:** Cálculo del gradiente vía *backpropagation* y actualización de pesos.
* **Entrenador (Trainer):** Bucle de entrenamiento sobre un dataset a lo largo de varias épocas.

---

## Arquitectura

```text
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
Regla de diseño clave: El número de neuronas de la última capa debe coincidir con la forma del valor que se quiere predecir. Como el dataset predice un único número (por ejemplo, un precio), la capa de salida tiene una sola neurona.

Estructura de archivos
Archivo	Responsabilidad
neurona.py	Clase Neuron: realiza z=w⋅x+b, aplica ReLU, guarda z y last_input para el backward.
layer.py	Clase Layer: ejecuta todas sus neuronas con la misma entrada y devuelve un vector de salidas.
network.py	Clase Network: encadena capas, pasando la salida de una como entrada de la siguiente.
activations.py	Función de activación ReLU.
losses.py	Función de pérdida MSE (Mean Squared Error).
optimizers.py	Clase Optimizer: calcula gradientes vía regla de la cadena (backpropagation) y actualiza pesos y bias.
trainer.py	Clase Trainer: bucle de entrenamiento sobre épocas y ejemplos del dataset.
datasets.py	Dataset de ejemplo: pares (x, real).
main.py	Punto de entrada: construye la red, entrena y ejecuta una predicción final.
Conceptos matemáticos aplicados
1. Forward pass
Cada neurona calcula:

z= 
i
∑
​
 w 
i
​
 x 
i
​
 +ba=ReLU(z)
2. Función de pérdida (MSE)
L=(y 
pred
​
 −y 
real
​
 ) 
2
 
3. Regla de la cadena (backpropagation)
La idea central del proyecto: para saber cómo un peso lejano de la salida afecta al error, se recorren los tramos intermedios y se multiplican sus derivadas.

∂w
∂L
​
 = 
∂a
∂L
​
 ⋅ 
∂z
∂a
​
 ⋅ 
∂w
∂z
​
 
Con las fórmulas concretas usadas en el proyecto:

∂y 
pred
​
 
∂L
​
 =2(y 
pred
​
 −y 
real
​
 ) 
∂z
∂a
​
 ={ 
1
0
​
  
z>0
z≤0
​
  
∂w 
i
​
 
∂z
​
 =x 
i
​
 
4. Propagación del error entre capas (deltas)
Para una neurona que no está en la capa de salida, su "delta" (sensibilidad al error) depende de los deltas de todas las neuronas de la capa siguiente a las que está conectada:

δ 
j
(l)
​
 =( 
k
∑
​
 δ 
k
(l+1)
​
 ⋅w 
k,j
(l+1)
​
 )⋅ 
∂z 
j
(l)
​
 
∂a 
j
(l)
​
 
​
 
Esto es lo que permite que el error, calculado solo al final de la red, se reparta hacia atrás por todas las capas anteriores — de ahí el nombre backpropagation.

5. Actualización de pesos (descenso de gradiente)
w 
nuevo
​
 =w−η⋅ 
∂w
∂L
​
 
donde η (lr) es la tasa de aprendizaje.

Flujo de ejecución
main.py crea las neuronas de cada capa con pesos y bias iniciales.

Trainer.train() recorre el dataset durante N épocas.

Para cada ejemplo (x, real):

Network.forward(x) propaga la entrada capa por capa.

Se calcula el loss con MSE.

Optimizer.update() ejecuta el backward: calcula los deltas desde la última capa hacia la primera y actualiza pesos y bias de cada neurona.

Al finalizar el entrenamiento, se ejecuta una predicción final con network.forward(...).

Detalle importante de la implementación: El entrenamiento se realiza en modalidad SGD puro (Stochastic Gradient Descent) — los pesos se actualizan después de cada ejemplo individual, no después de promediar varios. Esto significa que los mismos pesos son "negociados" secuencialmente por todos los ejemplos del dataset en cada época.

Resultados observados
Con el dataset de 4 ejemplos y lr=0.001 durante 100 épocas:

Ejemplo (x)	Valor real	Loss inicial	Loss final (época 100)
[2, 1]	7	~0.77	~2.05 (estabilizado)
[3, 2]	11	~3.23	~2.54
[4, 3]	15	~7.07	~0.61
[3.5, 2]	8	~36.0	~11.47 (no converge)
Observación clave: El ejemplo [3.5, 2] → 8 no logra reducir su error de forma consistente, mientras que los otros tres sí. Esto se debe a que, con solo 4 puntos de datos, ese ejemplo no es compatible con el mismo patrón lineal que explica a los otros tres — los pesos compartidos llegan a un punto de compromiso, no a una solución perfecta para los cuatro.

Conclusión validada: El mecanismo de forward + loss + backpropagation multicapa funciona correctamente (los gradientes se calculan y aplican según la regla de la cadena). La falta de convergencia total es una limitación de los datos, no del algoritmo.

Bugs encontrados y corregidos
Durante el desarrollo se identificaron y corrigieron los siguientes errores, documentados aquí por su valor didáctico:

Layer.forward sin return: El método calculaba el vector de salida pero no lo devolvía, provocando que la capa siempre entregara None.

Network.forward no encadenaba las capas: Se llamaba a layer.forward(x) con la entrada original en cada iteración, en lugar de layer.forward(entrada) con la salida acumulada de la capa anterior.

Desajuste de dimensiones silencioso (zip): Al pasar una entrada de 3 valores a neuronas con solo 2 pesos, zip() truncó el vector sin lanzar ningún error, produciendo un resultado numéricamente "válido" pero incorrecto.

Última capa con más de una neurona: Al tener 2 neuronas en la capa de salida, la red devolví un vector [a, b] que no podía compararse contra un real escalar. Se corrigió reduciendo la capa de salida a 1 neurona.

Variable de bucle reutilizando un nombre reservado: for x, self.weights in zip(self.weights, x) sobrescribía self.weights (la lista de pesos del objeto) con un valor numérico individual en cada iteración, corrompiendo el estado de la neurona para llamadas posteriores.

Falta de last_input en la neurona: El backward necesita conocer la entrada que recibió cada neurona en su último forward para calcular  
∂w 
i
​
 
∂z
​
 =x 
i
​
 ; no se guardaba inicialmente.

Optimizer.update diseñado para una sola neurona: La primera versión no contemplaba múltiples capas ni múltiples neuronas por capa; fue reescrita para calcular y propagar deltas por capa, de atrás hacia adelante.

Inconsistencia de nombres de atributo (nueronas vs neuronas): Error de tipeo que provocó un AttributeError al no coincidir con el atributo real de la clase Layer.

Limitaciones actuales
Este proyecto implementa el núcleo algorítmico de una red neuronal, pero carece de componentes necesarios para un caso de uso real (por ejemplo, predicción de precios de vivienda):

Dataset mínimo (4 ejemplos): Insuficiente para generalizar patrones reales.

Sin separación entrenamiento/prueba (train/test split): No hay forma de validar si la red generaliza a datos no vistos.

Sin normalización de entradas: Variables con escalas muy distintas afectarían negativamente el entrenamiento.

Entrenamiento SGD puro únicamente: Sin soporte para mini-batch, que es el estándar en la práctica profesional.

Sin regularización (weight decay, dropout): Riesgo de sobreajuste con datasets más grandes.

Optimizador básico: Sin momentum ni tasas de aprendizaje adaptativas (Adam, RMSprop).

Riesgo de neuronas "muertas": Con ReLU y pocas neuronas por capa, una neurona que cae permanentemente en zona negativa deja de aprender.

Próximos pasos
[ ] Añadir soporte para entrenamiento por mini-batches.

[ ] Implementar un módulo de división de datos (train/test split).

[ ] Agregar técnicas de normalización para las entradas.

[ ] Desarrollar e integrar el optimizador Adam.




Gemini es una IA y puede cometer errores, incluso sobre personas. Tu privacidad y GeminiSe abre en una ventana nueva

Analizando
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

Construir, sin librerías externas de *deep learning*, los componentes mínimos necesarios para que una red neuronal aprenda:

* **Neurona:** Unidad básica que combina entradas, pesos y *bias*.
* **Capa (Layer):** Conjunto de neuronas en paralelo que comparten la misma entrada.
* **Red (Network):** Secuencia de capas encadenadas (la salida de una alimenta a la siguiente).
* **Optimizador:** Cálculo del gradiente vía *backpropagation* y actualización de pesos.
* **Entrenador (Trainer):** Bucle de entrenamiento sobre un dataset a lo largo de varias épocas.

---

## Arquitectura

```text
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

> **Regla de diseño clave:** El número de neuronas de la última capa debe coincidir con la forma del valor que se quiere predecir. Como el dataset predice un único número (por ejemplo, un precio), la capa de salida tiene una sola neurona.

---

## Estructura de archivos

| Archivo | Responsabilidad |
| :--- | :--- |
| `neurona.py` | Clase `Neuron`: realiza $z = w \cdot x + b$, aplica ReLU, guarda $z$ y `last_input` para el *backward*. |
| `layer.py` | Clase `Layer`: ejecuta todas sus neuronas con la misma entrada y devuelve un vector de salidas. |
| `network.py` | Clase `Network`: encadena capas, pasando la salida de una como entrada de la siguiente. |
| `activations.py` | Función de activación ReLU. |
| `losses.py` | Función de pérdida MSE (*Mean Squared Error*). |
| `optimizers.py` | Clase `Optimizer`: calcula gradientes vía regla de la cadena (*backpropagation*) y actualiza pesos y *bias*. |
| `trainer.py` | Clase `Trainer`: bucle de entrenamiento sobre épocas y ejemplos del dataset. |
| `datasets.py` | Dataset de ejemplo: pares `(x, real)`. |
| `main.py` | Punto de entrada: construye la red, entrena y ejecuta una predicción final. |

---

## Conceptos matemáticos aplicados

### 1. Forward pass
Cada neurona calcula:

$$z = \sum_i w_i x_i + b \qquad a = 	ext{ReLU}(z)$$

### 2. Función de pérdida (MSE)

$$L = (y_{pred} - y_{real})^2$$

### 3. Regla de la cadena (backpropagation)
La idea central del proyecto: para saber cómo un peso lejano de la salida afecta al error, se recorren los tramos intermedios y se multiplican sus derivadas.

$$rac{\partial L}{\partial w} = rac{\partial L}{\partial a} \cdot rac{\partial a}{\partial z} \cdot rac{\partial z}{\partial w}$$

Con las fórmulas concretas usadas en el proyecto:

$$rac{\partial L}{\partial y_{pred}} = 2(y_{pred} - y_{real}) \qquad rac{\partial a}{\partial z} =  egin{cases} 1 & z > 0 \ 0 & z \le 0 \end{cases} \qquad rac{\partial z}{\partial w_i} = x_i$$

### 4. Propagación del error entre capas (deltas)
Para una neurona que no está en la capa de salida, su "delta" (sensibilidad al error) depende de los deltas de todas las neuronas de la capa siguiente a las que está conectada:

$$\delta_j^{(l)} = \left( \sum_{k} \delta_k^{(l+1)} \cdot w_{k,j}^{(l+1)} 
ight) \cdot rac{\partial a_j^{(l)}}{\partial z_j^{(l)}}$$

Esto es lo que permite que el error, calculated solo al final de la red, se reparta hacia atrás por todas las capas anteriores — de ahí el nombre *backpropagation*.

### 5. Actualización de pesos (descenso de gradiente)

$$w_{nuevo} = w - \eta \cdot rac{\partial L}{\partial w}$$

donde $\eta$ (`lr`) es la tasa de aprendizaje.

---

## Flujo de ejecución

1. `main.py` crea las neuronas de cada capa con pesos y *bias* iniciales.
2. `Trainer.train()` recorre el dataset durante $N$ épocas.
3. Para cada ejemplo `(x, real)`:
   * `Network.forward(x)` propaga la entrada capa por capa.
   * Se calcula el *loss* con MSE.
   * `Optimizer.update()` ejecuta el *backward*: calcula los deltas desde la última capa hacia la primera y actualiza pesos y *bias* de cada neurona.
4. Al finalizar el entrenamiento, se ejecuta una predicción final con `network.forward(...)`.

> **Detalle importante de la implementación:** El entrenamiento se realiza en modalidad **SGD puro** (*Stochastic Gradient Descent*) — los pesos se actualizan después de cada ejemplo individual, no después de promediar varios. Esto significa que los mismos pesos son "negociados" secuencialmente por todos los ejemplos del dataset en cada época.

---

## Resultados observados

Con el dataset de 4 ejemplos y $	ext{lr} = 0.001$ durante 100 épocas:

| Ejemplo (`x`) | Valor real | Loss inicial | Loss final (época 100) |
| :--- | :--- | :--- | :--- |
| `[2, 1]` | 7 | ~0.77 | ~2.05 (estabilizado) |
| `[3, 2]` | 11 | ~3.23 | ~2.54 |
| `[4, 3]` | 15 | ~7.07 | ~0.61 |
| `[3.5, 2]` | 8 | ~36.0 | ~11.47 (no converge) |

* **Observación clave:** El ejemplo `[3.5, 2]` $
ightarrow$ `8` no logra reducir su error de forma consistente, mientras que los otros tres sí. Esto se debe a que, con solo 4 puntos de datos, ese ejemplo no es compatible con el mismo patrón lineal que explica a los otros tres — los pesos compartidos llegan a un punto de compromiso, no a una solución perfecta para los cuatro.
* **Conclusión validada:** El mecanismo de *forward* + *loss* + *backpropagation* multicapa funciona correctamente (los gradientes se calculan y aplican según la regla de la cadena). La falta de convergencia total es una limitación de los datos, no del algoritmo.

---

## Bugs encontrados y corregidos

Durante el desarrollo se identificaron y corrigieron los siguientes errores, documentados aquí por su valor didáctico:

* **`Layer.forward` sin `return`:** El método calculaba el vector de salida pero no lo devolvía, provocando que la capa siempre entregara `None`.
* **`Network.forward` no encadenaba las capas:** Se llamaba a `layer.forward(x)` con la entrada original en cada iteración, en lugar de `layer.forward(entrada)` con la salida acumulada de la capa anterior.
* **Desajuste de dimensiones silencioso (`zip`):** Al pasar una entrada de 3 valores a neuronas con solo 2 pesos, `zip()` truncó el vector sin lanzar ningún error, produciendo un resultado numéricamente "válido" pero incorrecto.
* **Última capa con más de una neurona:** Al tener 2 neuronas en la capa de salida, la red devolvía un vector `[a, b]` que no podía compararse contra un real escalar. Se corrigió reduciendo la capa de salida a 1 neurona.
* **Variable de bucle reutilizando un nombre reservado:** `for x, self.weights in zip(self.weights, x)` sobrescribía `self.weights` (la lista de pesos del objeto) con un valor numérico individual en cada iteración, corrompiendo el estado de la neurona para llamadas posteriores.
* **Falta de `last_input` en la neurona:** El *backward* necesita conocer la entrada que recibió cada neurona en su último *forward* para calcular $rac{\partial z}{\partial w_i} = x_i$; no se guardaba inicialmente.
* **`Optimizer.update` diseñado para una sola neurona:** La primera versión no contemplaba múltiples capas ni múltiples neuronas por capa; fue reescrita para calcular y propagar deltas por capa, de atrás hacia adelante.
* **Inconsistencia de nombres de atributo (`nueronas` vs `neuronas`):** Error de tipeo que provocó un `AttributeError` al no coincidir con el atributo real de la clase `Layer`.

---

## Limitaciones actuales

Este proyecto implementa el núcleo algorítmico de una red neuronal, pero carece de componentes necesarios para un caso de uso real (por ejemplo, predicción de precios de vivienda):

* **Dataset mínimo (4 ejemplos):** Insuficiente para generalizar patrones reales.
* **Sin separación entrenamiento/prueba (*train/test split*):** No hay forma de validar si la red generaliza a datos no vistos.
* **Sin normalización de entradas:** Variables con escalas muy distintas afectarían negativamente el entrenamiento.
* **Entrenamiento SGD puro únicamente:** Sin soporte para *mini-batch*, que es el estándar en la práctica profesional.
* **Sin regularización (*weight decay*, *dropout*):** Riesgo de sobreajuste con datasets más grandes.
* **Optimizador básico:** Sin *momentum* ni tasas de aprendizaje adaptativas (Adam, RMSprop).
* **Riesgo de neuronas "muertas":** Con ReLU y pocas neuronas por capa, una neurona que cae permanentemente en zona negativa deja de aprender.
