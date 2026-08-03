#Importamos trainer para ejecutar
from trainer import Trainer
from neurona import Neuron
from optimizers import Optimizer
from datasets import dataset

neurona = Neuron(weights=[2,3],bias=1)

optimizador = Optimizer(step=0.001)

trainer = Trainer()

trainer.train(
    neurona,
    dataset,
    optimizador
)
