from random import randint

import torch
from problog.logic import Constant

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


def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 18))])
    return new_query


dataset = MNISTOperator(
    dataset_name="train",
    function_name="addition_noisy",
    operator=sum,
    size=1,
)

noisy_dataset = MutatingDataset(dataset, NoiseMutatorDecorator(0.4, noise))
#print(noisy_dataset)
queries = DataLoader(noisy_dataset, 1)

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
model = Model("models/noisy_addition.pl", [net])
#model = Model("models/noisy_single.pl", [net])

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

model.set_engine(ExactEngine(model))
model.optimizer = SGD(model, 1e-3)

train = train_model(model, queries, 10, log_iter=100)

""" train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name) """

if __name__=="__main__":
    print(model.networks["mnist_net"])
    #print(MNIST_train[[0]])
    #print(len(i1[0]))
    #print(model.networks['mnist_net'].network_module(i1[0]))
    pass