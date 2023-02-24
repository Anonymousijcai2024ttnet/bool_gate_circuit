# TTnet as a Boolean Logic Gate Circuit

## Overview

This repository contains the results of Table 5.

## Results

The following datasets have been evaluated:

- MNIST
- CIFAR10

### MNIST

The performances are:

|       | Accuracy | Total number of gates for the filter | Size Circuit (OP) | 
|-------|:--------:|:------------------------------------:|:-----------------:|
| Small |  97.44%  |                 477                  |        37K        |    
| Big   |  98.39%  |                 2694                 |       203K        | 

### CIFAR10
The performances are:

|       | Accuracy | Total number of gates for the filter | Size Circuit (OP) | 
|-------|:--------:|:------------------------------------:|:-----------------:|
| Small |  54.49%  |                 3469                 |       804K        |    



## Usage

### Configuration
This project uses Python 3.10. To install the required packages, run the following command:

```
pip3 install -r requirements.txt
```

### Running Inference

In progress



### Changing the Dataset

To change the dataset, modify the dataset field in the `config.yaml` file.



