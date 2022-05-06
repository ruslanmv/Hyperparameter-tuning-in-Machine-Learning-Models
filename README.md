#  Hyperparameter tuning in Machine Learning Models

Parameters which define the model architecture are referred to as hyperparameters and thus this process of searching for the ideal model architecture is referred to as hyperparameter tuning.

In machine learning, we use the term hyperparameter to distinguish from standard model parameters. So, it is worth to first understand what those are.

 A machine learning model is the definition of a mathematical formula with a number of parameters that need to be learned from the data. That is the crux of machine learning: fitting a model to the data. 

This is done through a process known as model training. In other words, by training a model with existing data, we are able to fit the model parameters. However, there is another kind of parameters that cannot be directly learned from the regular training process. These parameters express “higher-level” properties of the model such as its complexity or how fast it should learn. They are called hy- perparameters. Hyperparameters are usually fixed before the actual training process begins. So, how are hyperparameters decided? That is probably beyond the scope of this question, but suffice to say that, broadly speaking, this is done by setting different values for those hyperparameters, training different models, and deciding which ones work best by testing them. So, to summarize hyperparameters define higher level concepts about the model such as complexity, or capacity to learn. Cannot be learned directly from the data in the standard model training process and need to be predefined. Can be decided by setting different values, training different models, and choosing the values that test better Some examples of hyperparameters: Number of leaves or depth of a tree Number of latent factors in a matrix factor- ization Learning rate (in many models) Number of hidden layers in a deep neural network Number of clusters in a k-means clustering

## 2.2 Tuning Approaches

Most common approaches for tuning Hyperparameters are: • Manual Approaches • Algorithmic Approaches

### 2.2.1 Manual Approaches

The straightforward way to tune hyperparameters is based on human expertise. Ex- perienced Machine Learning practitioners know approximately how to choose good hyperparameters. For a new dataset, they will follow a trial and error process with various configurations of hyperparameters. This process of experimenting with hy- perparameters is heuristic and different people with different experience might come up with different settings and the process is not easily reproducible. There is also a real risk that a human won’t achieve a near optimum setting of hyperparameters. Humans are not good at handling high dimensional data and can easily misinterpret or miss trends and relationships when trying to tune multiple hy- perparameters. For instance, while tuning just two parameters, practitioners often fall back to tuning one parameter then tuning the second parameter. This may lead to concluding improvement in performance has plateaued while adjusting the second hyperparameter, while more improvement might be available by going back to chang- ing the first hyperparameter. This difficulty requires an automatic and reproducible approach for hyperparameter tuning. In practice, an overly complex model family does not necessarily include the target function or the true data-generating process, or even a close approximation of either. We almost never have access to the true data-generating process so we can never know for sure if the model family being estimated includes the generating process or not. Most applications of deep learning algorithms, however,are to domains where the true data-generating process is almost certainly outside the model family. Deep learning algorithms are typically applied to extremely complicated domains such as images, audio sequences and text, for which the true generation process essentially involves simulating the entire universe. To some extent, we are always trying to fit a square peg (the data-generating process) into a round hole (our model family).

### 2.2.2 Algorithm Approaches

We can generalize this problem as: given a function that accepts inputs and returns a numerical output, how can we efficiently find the inputs, or parameters, that maximize the function’s output? In the case of hyperparameter tuning, the input to our function is the hyperpa- rameters of the model, and the output is the result of measuring the corresponding model’s performance on an offline dataset, a common practice in machine learning. The inputs of the function are parameters, and the parameter space represents all possible values of the parameters, usually defined as “acceptable” bounds for each parameter. The dimension of the function is the number of these parameters. An optimization strategy defines how we will select parameters and how many function evaluations we will perform for every optimization. The final result of an

## 2.2 Tuning Approaches

Optimization is the “winning” set of parameters that have produced the highest output observed for our function. Considerations when evaluating the performance of an optimization strategy include: What is the best output value observed by this strategy?What is the expected cost of this strategy? How many times do I have to evaluate the function to achieve the best observed value? We’ll explore three different methods of optimizing hyperparameters: grid search, random search, and Bayesian optimization. There are other potential strategies, but many of these require too many function evaluations per optimization to be feasible.

### Grid Search

In grid search, we try a set of configurations of hyperparameters and train the algorithm accordingly, choosing the hyperparameter configuration that gives the best performance. In practice, practitioners specify the bounds and steps between values of the hyperparameters, so that it forms a grid of configurations. Practitioners typ- ically start with a limited grid with relatively large steps between parameter values, then extend or make the grid finer at the best configuration and continue searching on the new grid. This process is called manual grid search. Grid search is a costly approach. Assuming we have n hyperparameters and each hyperparameter has two values, then the total number of configurations is 2n. Therefore it is only feasible to do grid search on a small number of configurations. Fortunately, grid search is embarrassingly parallel, meaning it can be parallelized easily where each worker works on different parameter settings. This makes grid search a bit more feasible given enough computational power.

### Random Search

In Random Search for Hyperparameter Optimization, the authors show a surprising result: by navigating the grid of hyperparameters randomly, one can obtain similar performance to a full grid search. The authors show that if the close-to-optimal region of hyperparameters occupies at least 5% of the search space, then a random search with a certain number of trials (typically 40-60 trials) will be able to find that region with high probability. Random search is surprisingly simple and effective, therefore it is considered by many practitioners as the de facto method for tuning hyperparameters. Like grid search, it is embarassingly parallel, yet the number of trials is much less, while the performance is comparable. For discrete problems in which no efficient solution method is known, it might be necessary to test each possibility sequentially in order to determine if it is the solution. Such exhaustive examination of all possibilities is known as exhaustive search, direct search, or the "brute force" method. Unless it turns out that NP-problems are equiv- alent to P-problems, which seems unlikely but has not yet been proved, NP-problems can only be solved by exhaustive search in the worst case.[wolfram]

## 2 Hyperparameters

### Automatic Hyperparameter Tuning

In Grid Search and Random Search, we try the configurations randomly and blindly. The next trial is independent to all the trials done before. In contrast, automatic hyperparameter tuning forms knowledge about the relation between the hyperpa- rameter settings and model performance in order to make a smarter choice for the next parameter settings. Typically it will first collect the performance at several configurations, then make some inference and decide what configuration to try next. The purpose is to minimize the number of trials while finding a good optimum. This process, by nature, is sequential and not easily parallelizable. It can be seen that hyperparameter tuning is an optimization problem, where we find the hyperparameter setting that maximizes the performance of the model on a validation set. However, the mapping from the hyperparameters to the perfor- mance on the validation set cannot be written into a formula, hence we don’t know the derivatives of this function – it’s known as a blackbox function. Because it’s a blackbox function, optimization techniques like Newton method or Gradient Descent cannot be applied. Most of the approaches for tuning hyperparameters fall into Sequential Model- based Global Optimization (SMBO). These approaches use a surrogate function to approximate the true blackbox function. Typically the inner loop of SMBO is the optimization of this surrogate, or some kind of transformation done on the surrogate. The configuration that maximizes this surrogate will be the one should be tried next. SMBO algorithms differ in the criteria by which it optimizes the surrogate, and in the way they model the surrogate given the observation history. Several SMBO approaches for hyperparameters have been recently proposed in literature:

### a) Bayesian Optimization

It uses a Gaussian Process to model the surrogate, and typically optimizes the Ex- pected Improvement, which is the expected probability that new trials will improve upon the current best observation. Gaussian Process is a distribution over functions. A sample from a Gaussian process is an entire function. Training a Gaussian Pro- cess involves fitting this distribution to the given data, so that it generates functions that are close to the observed data. Using Gaussian process, one can compute the Expected Improvement of any point in the search space. The one gives the high- est expected improvement will be tried next. Bayesian Optimization typically gives non-trivial, off-the-grid values for continuous hyperparameters (like the learning rate, regularization coefficient,...) and was shown to beat human performance on some good benchmark datasets. A well-known implementation of Bayesian Optimization is Spearmint.

### b) Sequential Model-based Algorithm Configuration (SMAC)

It uses a random forest of regression trees to model the objective function, new points are sampled from the region considered optimal (high Expected Improvement) by the random forest.

### c) Tree-structured Parzen Estimator (TPE)

It is an improved version of SMAC, where two separated models are used to model the posterior. A well-known implementation of TPE is hyperopt.



There are several tools and packages available for the task. [Here](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) is a good paper on the topic.

- [hyperopt](https://hyperopt.github.io/hyperopt/) implements random search and tree of parzen estimators optimization.
- [Scikit-Optimize](https://scikit-optimize.github.io/) implements a few others, including Gaussian process Bayesian optimization.
- [SigOpt](https://sigopt.com/) is a convenient service (paid, although with a free tier and extra allowance for students and researchers) for hyperparameter optimization. It is based upon Yelp's [MOE](https://github.com/Yelp/MOE), which is open source (although the published version doesn't seem to update much) and can, in theory, be used on its own, although it would take some additional effort.
- [Spearmint](https://github.com/HIPS/Spearmint) is a commonly referred package too, also open source but not free for commercial purposes (although you can fall back to a [less restrictive older version](https://github.com/JasperSnoek/spearmint)). It looks good, but not very active, and the available version is not compatible with Python 3 (even though pull requests have been submitted to fix that).
- [BayesOpt](https://bitbucket.org/rmcantin/bayesopt) seems to be the golden standard in Bayesian optimization, but it's mainly C++, and the Python interface doesn't look very documented.



## Most common Libraries of Hyperparameter tunning



|                       Library Name                        |                         Description                          | Framework |
| :-------------------------------------------------------: | :----------------------------------------------------------: | :-------: |
| [Keras Tuner]( https://github.com/keras-team/keras-tuner) | A hyperparameter tuner for Keras, specifically for tf.keras with TensorFlow 2.0. |  `Keras`  |
|       [talos]( https://github.com/autonomio/talos)        | Talos radically changes the ordinary Keras workflow by fully automating hyperparameter tuning and model evaluation. |  `Keras`  |
|    [hyperas]( https://github.com/maxpumperla/hyperas)     | A very simple convenience wrapper around hyperopt for fast prototyping with keras models. Hyperas lets you use the power of hyperopt without having to learn the syntax of it. |  `Keras`  |


|                       Library Name                        |                         Description                          | Framework |
| :-------------------------------------------------------: | :----------------------------------------------------------: | :-------: |
|  [Auto-PyTorch](https://github.com/automl/Auto-PyTorch)   | This a very early pre-alpha version of our upcoming Auto-PyTorch. So far, Auto-PyTorch supports featurized data (classification, regression) and image data (classification). | `PyTorch` |
| [hypersearch]( https://github.com/kevinzakka/hypersearch) | une the hyperparameters of your PyTorch models with HyperSearch. | `PyTorch` |
|      [botorch]( https://github.com/pytorch/botorch)       | BoTorch is a library for Bayesian Optimization built on PyTorch. | `PyTorch` |

|                         Library Name                         |                         Description                          | Framework |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-------: |
|        [tpot]( https://github.com/EpistasisLab/tpot)         | TPOT stands for Tree-based Pipeline Optimization Tool. Consider TPOT your Data Science Assistant. TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. | `General` |
|           [nni]( https://github.com/microsoft/nni)           | NNI (Neural Network Intelligence) is a lightweight but powerful toolkit to help users automate Feature Engineering, Neural Architecture Search, Hyperparameter Tuning and Model Compression. | `General` |
|      [xcessiv]( https://github.com/reiinakano/xcessiv)       | Xcessiv holds your hand through all the implementation details of creating and optimizing stacked ensembles so you're free to fully define only the things you care about. | `General` |
|          [ray]( https://github.com/ray-project/ray)          | Ray is a fast and simple framework for building and running distributed applications. | `General` |
| [tune-sklearn]( https://github.com/ray-project/tune-sklearn) | Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers. | `General` |
|         [optuna]( https://github.com/optuna/optuna)          | Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. | `General` |
|      [Hyperopt]( https://github.com/hyperopt/hyperopt)       | Hyperopt is a Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions. | `General` |
| [scikit-optimize]( https://github.com/scikit-optimize/scikit-optimize) | Scikit-Optimize, or skopt, is a simple and efficient library to minimize (very) expensive and noisy black-box functions. | `General` |
|            [Ax]( https://github.com/facebook/Ax)             | Ax is an accessible, general-purpose platform for understanding, managing, deploying, and automating adaptive experiments. | `General` |
|       [Spearmint]( https://github.com/HIPS/Spearmint)        | Spearmint is a software package to perform Bayesian optimization. | `General` |
| [hyperparameter_hunter]( https://github.com/HunterMcGushion/hyperparameter_hunter) | Automatically save and learn from Experiment results, leading to long-term, persistent optimization that remembers all your tests. | `General` |
|        [sherpa]( https://github.com/sherpa-ai/sherpa)        |         A Python Hyperparameter Optimization Library         | `General` |
| [auptimizer]( https://github.com/LGE-ARC-AdvancedAI/auptimizer) | Auptimizer is an optimization tool for Machine Learning (ML) that automates many of the tedious parts of the model building process. | `General` |
|      [advisor]( https://github.com/tobegit3hub/advisor)      | Advisor is the hyper parameters tuning system for black box optimization. | `General` |
|   [test-tube](https://github.com/williamFalcon/test-tube )   | Test tube is a python library to track and parallelize hyperparameter search for Deep Learning and ML experiments. | `General` |
|  [Determined]( https://github.com/determined-ai/determined)  | Determined helps deep learning teams train models more quickly, easily share GPU resources, and effectively collaborate. | `General` |



Another interesting Libraries

https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb



