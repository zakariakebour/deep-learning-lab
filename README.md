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
