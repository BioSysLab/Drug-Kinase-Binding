# Learn with confidence
Predicting an uncertainty estimation alongside with the point prediction in deep learning.

## Introduction
The problem of uncertainty prediction in machine learning is very important although not very
well-known. We made an effort to predict uncertainties associated with the point forecasts for our deep graph convolutional neural network, following three approaches.

## “Distance” from the training data 
In this approach we investigate if the position of the input data point (protein-molecule pair representation) in the learned latent space could work as a measure of uncertainty.
On this front, in the 128-dimensional latent space of our deep graph convolutional neural network we calculated all the euclidean distances between every validation and training point for a random-split validation set. For estimating the uncertainty of a validation point prediction we followed three approaches. First, for a given validation point, we counted its “neighbors” in the 128-dimensional sphere of a finite radius and considered the inverse of this number as a metric of uncertainty. Second, we calculated uncertainty as the normalized median of all the distances between this point and the training set. Third, we did not use any distances but calculated the uncertainty as the level of the point prediction error of the neighboring training points in the latent space. We assumed that validation points that are included in areas of high error in the latent space should also be accompanied with a high uncertainty prediction. The latter way was preoved to be also the best.

## Test-time Dropout
Test-time dropout which is also known as Bayesian deep learning has been proved to be a variational approximation to a Gaussian process and therefore uncertainty estimates using the variation of the different predictions can be extracted [1]. It can be proven that dropout neural networks are identical to variational inference in Gaussian processes (variational inference is an approach of approximating model posterior distribution which would otherwise be difficult to work with directly). Actually, averaging forward passes through the network (which is another way to say test-time dropout), is equivalent to Monte Carlo integration over a Gaussian process posterior approximation.

## Ensemble Learning
To predict confidence intervals out of ensemble learning, one can take the standard deviation of the different models predictions as a metric of uncertainty. we implemented the uncertainty estimation using an ensemble of deep models trained on the same data set (108 slightly different graph convolutional neural networks as for their architecture configuration) that predict two values instead of one. To be more specific, we tried to recreate [a state of the art model] [2] that was published by Google’s DeepMind in 2017 and constitutes the best approach for estimating uncertainties in deep learning according to the author’s opinion. We use a network that outputs two values in the final layer, corresponding to the predicted mean μ(x) and variance σ(x) > 0. 

## Validate Uncertainties
Naturally, when trying to develop a model that apart from the point predictions also outputs uncertainty estimations, one must have a validation metric to judge these uncertainty predictions after the training. We decided that the validation of the uncertainty estimates can be calculated using the absolute error of the point predictions (i.e. the mean of all the predictions for an input), expecting the data points with the larger error to also have the maximum uncertainty. That way, the validation accuracy is defined as the correlation between the predicted uncertainties and the error of the correspondent inputs.

Some results on the pearson correlation between the absolute error and the uncertainties for the three ways mentioned above are shown in the pictures.


## References

[1] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model
uncertainty in deep learning, 2015.

[2] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable pre-
dictive uncertainty estimation using deep ensembles, 2016.
