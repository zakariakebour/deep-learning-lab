# Importamos trainer para ejecutar
from trainer import Trainer
from neurona import Neuron
from optimizers import Optimizer
from datasets import dataset
from layer import Layer
from network import Network

# Neuronas de la capa 1
neurona_1_capa1 = Neuron(weights=[2,3],bias=1,z=None)
neurona_2_capa1 = Neuron(weights=[1,4],bias=2,z=None)

# Neuronas de la capa 2
neurona_1_capa2 = Neuron(weights=[4,2],bias=1,z=None)
neurona_2_capa2 = Neuron(weights=[3,1],bias=-1,z=None)

optimizador = Optimizer(lr=0.001)


layer1 = Layer(neuronas=[neurona_1_capa1,neurona_2_capa1])
layer2 = Layer(neuronas=[neurona_1_capa2,neurona_2_capa2])

# Red que úne las cápas
network = Network([
    layer1,
    layer2
])

# Para entrenar las neuronas
trainer = Trainer()

# Trainer recibe la red, dataset y el optimizador
trainer.train(
    network, 
    dataset,
    optimizador
)

resultado = network.forward([3.5,2])
print(resultado) 