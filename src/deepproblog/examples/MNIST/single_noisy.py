from random import randint

import torch
from problog.logic import Constant
import sys
from deepproblog.dataset import DataLoader
from deepproblog.dataset import NoiseMutatorDecorator, MutatingDataset
from deepproblog.engines import ExactEngine
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test, addition, single_num
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.query import Query
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

# Noise function to pass to mutator
def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 9))])
    return new_query



# Size of dataset
#N=5000
epoch_count=1
# Set fraction of noise
noise_fraction=0.8
dataset=single_num(1,"train")#.subset(N)


noisy_dataset=MutatingDataset(dataset,NoiseMutatorDecorator(noise_fraction,noise))
queries=DataLoader(noisy_dataset,1)
test_set=single_num(1,"test")

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
model = Model("models/noisy_single.pl", [net])

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

model.set_engine(ExactEngine(model))
model.optimizer = SGD(model, 3e-3)

train = train_model(model, queries, epoch_count, log_iter=100)
get_confusion_matrix(model, test_set, verbose=1).accuracy()
""" train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name) """

if __name__=="__main__":
    
    pass