from collections import namedtuple
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from deepproblog.examples.MNIST.data import addition, single_num, MNIST_train, create_noisy_training_data
from deepproblog.utils.logger import Logger
from deepproblog.utils.stop_condition import StopOnPlateau
#from deepproblog.examples.MNIST.neural_baseline.baseline_models import Separate_Baseline, Single_Baseline
from baseline_models import Separate_Baseline, Single_Baseline


""" def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 18))])
    return new_query """

def test_addition(dset):
    confusion = np.zeros(
        (19, 19), dtype=np.uint32
    )  # First index actual, second index predicted
    correct = 0
    n = 0
    for i1, i2, l in dset:
        i1 = i1[0]
        i2 = i2[0]
        i1 = Variable(i1.unsqueeze(0))
        i2 = Variable(i2.unsqueeze(0))
        outputs = net.forward(i1, i2)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print("Accuracy: ", acc)
    return acc

def test_single(dset):
    confusion = np.zeros(
        (10, 10), dtype=np.uint32
    )  # First index actual, second index predicted
    correct = 0
    n = 0
    for image, l in dset:
        image = Variable(image.unsqueeze(0))
        outputs = net.forward(image)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    #print("Accuracy: ", acc)
    return acc


test_dataset = single_num(1, "test")

if __name__ == "__main__":
    pass
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    Train = namedtuple("Train", ["logger"])
    model, modelname = Single_Baseline, "Separate"
    
    # Set required fraction of noise
    noise=0.8
    noisy_MNIST=create_noisy_training_data(noise)
    # for N in [50, 100, 200, 500, 1000]:
    accuracies=[]
    for k in range(4):
        print("Round:", k+1)
        for N in [5000]:
            train_dataset = single_num(1, "noise").subset(N)
            val_dataset = single_num(1, "noise").subset(N, N + 100)
            for batch_size in [4]:
                test_period = N // batch_size
                log_period = N // (batch_size * 10)
                trainloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                running_loss = 0.0
                log = Logger()
                i = 1
                net = model(batched=True, probabilities=False)
                optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)
                criterion = nn.CrossEntropyLoss()
                stop_condition = StopOnPlateau("Accuracy", patience=5)
                train_obj = Train(log)
                j = 1
                while not stop_condition.is_stop(train_obj):
                    print("\rEpoch {}...".format(j), end="",flush=True)
                    for image, l in trainloader:
                        image, l = Variable(image),Variable(l)
                        optimizer.zero_grad()
                        outputs = net(image)
                        loss = criterion(outputs, l)
                        loss.backward()
                        optimizer.step()
                        running_loss += float(loss)
                        if i % log_period == 0:
                            """ print(
                                "Iteration: ",
                                i,
                                "\tAverage Loss: ",
                                running_loss / log_period,
                            ) """
                            log.log("loss", i, running_loss / log_period)
                            running_loss = 0
                        if i % test_period == 0:
                            log.log("Accuracy", i, test_single(val_dataset))
                        i += 1
                    j += 1
                """ torch.save(
                    net.state_dict(), "../models/pretrained/addition_{}.pth".format(N)
                ) """
                #print()
                acc=test_single(test_dataset)
                print("Accuracy:",acc)
                print("-"*10)
                
                accuracies.append(acc)
                log.comment("Accuracy\t{}".format(acc))
    print(accuracies,np.mean(accuracies),np.std(accuracies))
