# Importamos trainer para ejecutar
from trainer import Trainer
from neurona import Neuron
from optimizers import Optimizer
from datasets import dataset
from layer import Layer

neurona = Neuron(weights=[2,3],bias=1,z=None)

neurona1 = Neuron(weights=[1,4],bias=2,z=None)
optimizador = Optimizer(lr=0.001)

trainer = Trainer()

trainer.train(
    neurona, 
    dataset,
    optimizador
)

layer = Layer(neuronas=[neurona,neurona1])

salida = layer.forward(x=[2,2])

print(f"Resultado de layer: {salida}")