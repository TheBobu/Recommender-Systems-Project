# Recommender Systems Project
***Group 20**: Ahmeti Ardit, Florescu Andrei, Xhetani Klea, Shazivari Xhilda, Kazazi Fjona*

## Table of contents

- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
    - [Collaborative filtering](#collaborative-filtering)
    - [Neural collaborative filtering](#neural-collaborative-filtering)
- [Project setup](#project-setup)
    - [Conda environment setup](#conda-environment-setup)
    - [Simple python setup](#simple-python-setup)
- [Running the project](#running-the-project)
    - [Running using Jupyter notebooks](#running-using-jupyter-notebooks)
    - [Running using the terminal](#running-using-the-terminal)
- [Testset results](#testset-results)


## Introduction

This repository contains two implementations of recommender systems for the [ACM RecSys 2024 Challenge](http://www.recsyschallenge.com/2024/).

### Collaborative filtering

The collaborative filtering model combines the predictions of a **user based CF model** and an **item based CF model**. The user based CF model uses an interaction matrix between users and articles which contains an aggregate between number of interactions with that article, read time and scroll percentage into one interaction score. The item based CF model uses also this interaction matrix, but the simialrity between articles is calculated based on the article metadata (topic, category, total page views, total read time, sentiment score). Then the final prediction is calculated as the average of these two interaction scores. The model implementation can be found in the [cf_model.py](./cf_model.py) file.


### Neural collaborative filtering

NCF framework integrates deep neural networks into collaborative filtering. It offers a multi-layer architecture where each layer processes user and item features. This method addresses the limitations of traditional matrix factorization techniques by learning user-item interaction functions through a multi-layer perceptron. It also provides enhanced flexibility and performance in recommendation systems.


## Project setup

To start using the project you need a python version installed or miniconda3. The packages required for this project can be found in the [requriements.txt](./requirements.txt) file.

### Conda environment setup

If you have conda installed on your machine, to create the environment required to run this project you need to run the command:

```sh
> conda env create -f environment.yml
```

This will create an environment called `recsys-env`

### Simple python setup

If you have only python installed without conda virtual environments you can install the required packages using the command:

```sh
> pip install -r requirements.txt
```

## Running the project

### Running using Jupyter notebooks

To run the project you can run the python files for each model or go to their associated notebooks:
- [cf_model_tests.ipynb](./cf_model_tests.ipynb) for the **collaborative filtering model**
- [main.ipynb](./main.ipynb) for the **neural collaborative filtering model**
- [nrms_ebnerd_final.ipynb](./nrms_ebnerd_final.ipynb) for the **baseline model**

Other notebooks include:
- [data_preparation.ipynb](./data_preparation.ipynb) for exploratory data analysis
- [make_beyond_accuracy.ipynb](./make_beyond_accuracy.ipynb) for beyond accuracy measurements

### Running using the terminal

Running the python file [cf_model.py](./cf_model.py) will generate predictions into a `.txt` file for the **simple collaborative filtering** model. To run the file use the command:

```sh
> python -m cf_model --data_folder <path/to/data> 
%% For example ./data/ebnerd_demo
```

If you need help with the parameters you can run the command:
```sh
> python -m cf_model -h
```

The example command can also be found in the [run.sh](./run.sh) file.

## Testset results

The results after predicting the test set using the neural collaborative filtering model can be found on jupyter hub under the link: ~/shared/194.035-2024S/groups/Gruppe_20