# DeepProbLog
This fork of https://github.com/ML-KULeuven/deepproblog/tree/master is solely for academic purposes.
## Installation
DeepProbLog can easily be installed using the following command:
Make sure the following packages are installed:
```
pip install deepproblog
```

## Test
To make sure your installation works, install pytest 
```
pip install pytest
````
and run 
```
python -m deepproblog test
```

## Requirements

DeepProbLog has the following requirements:
* Python > 3.9
* [ProbLog](https://dtai.cs.kuleuven.be/problog/)
* [PySDD](https://pysdd.readthedocs.io/en/latest/)
* [PyTorch](https://pytorch.org/)
* [TorchVision](https://pytorch.org/vision/stable/index.html)

## Inference

To use Inference, we have the following additional requirements
* [PySwip](https://github.com/ML-KULeuven/pyswip) 
    - Use `pip install git+https://github.com/ML-KULeuven/pyswip`
* [SWI-Prolog < 9.0.0](https://www.swi-prolog.org/)
The latter can be installed on Ubuntu with the following commands:
```
sudo apt-add-repository ppa:swi-prolog/stable
sudo apt install swi-prolog=8.4* swi-prolog-nox=8.4* swi-prolog-x=8.4*
```
## Experiments

The experiments are presented in the report are available in the [src/deepproblog/examples/MNIST](src/deepproblog/examples/MNIST) directory.

Namely the script [single_noisy.py](src/deepproblog/examples/MNIST/single_noisy.py) and the script [neural_baseline/baseline.py](src/deepproblog/examples/MNIST/neural_baseline/baseline.py).

The DeepProblog Program can be found in [noisy_single.pl](src\deepproblog\examples\MNIST\models\noisy_single.pl)

## Papers Directly Used
1. Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, Luc De Raedt:
*DeepProbLog: Neural Probabilistic Logic Programming*. NeurIPS 2018: 3753-3763 ([paper](https://papers.nips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html))
2. Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, Luc De Raedt:
*Neural Probabilistic Logic Programming in DeepProbLog*. AIJ ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0004370221000552))
3. Robin Manhaeve, Giuseppe Marra, Luc De Raedt:
*Approximate Inference for Neural Probabilistic Logic Programming*. KR 2021

