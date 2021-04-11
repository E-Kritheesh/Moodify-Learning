This is the report of the google machine learning crash course(upto regularisation: sparsity).

# ML Concepts

## Framing

This topic introduces the basic ML terms such as label, features, model, training, inference that one needs to know. 
A comparison between regression model(predicts continuous values) and classification model(predicts discrete values) is also made.
Assessment of check your understanding questions of this topic: 

![ML Framing](https://user-images.githubusercontent.com/81472530/114296659-203ef400-9aca-11eb-9f03-85fae3c22271.jpg)

![ML Framing2](https://user-images.githubusercontent.com/81472530/114296663-259c3e80-9aca-11eb-8fcd-0f655246e408.jpg)


## Descending into ML

### Linear regression model

Using an example where features and label can be approximately related as a linear function, a model y' = wx + b is used to describe the relationship.
The training of the model to come up with an optimal value of weight vector(w) and bias(b) is explained.
As the data cannot be perfectly expressed as a line equation, there is a need to minimise the error between the predicted value and the observation.
The mean squared loss function, which is convex hence can be minimised, proves to be a practical loss function.
Assessment of check your understanding questions of this topic:

![ML Descending into ML](https://user-images.githubusercontent.com/81472530/114300403-cd6f3780-9add-11eb-9251-70abb1a6e141.jpg)


## Reducing loss

This topic focuses on reducing loss using an iterative approach. The squared loss value gets reduced in every iteration until it converges to the lowest possible loss.
Gradient descent mechanism is an effective way to minimise the error in lesser number of iterations.
A random initial value is picked and the next weight is chosen in the direction of the negative gradient to approach the converging point faster.
Gradient descent algorithms multiply the gradient by the hyperparameter: learning rate to optimise the iterations taken to achieve the minimum loss.
The conclusion from the playground exercise was that a smaller learning rate takes much longer time to converge and a large learning rate may not converge at all(climbing the curve instead of descending to the bottom).
It is also found that performing gradient descent on a small batch is usually more efficient than the full batch.
Assessment of check your understanding questions of this topic:

![ML Reducing Loss](https://user-images.githubusercontent.com/81472530/114301117-ec22fd80-9ae0-11eb-9628-5953f62160ea.jpg)


## First steps with TF

This topic is an introduction to tensor flow.
It mainly focuses on tuning hyperparameters like batch size, learning rate, epochs and plotting the corresponding loss curves for a synthetic followed by a real data.
The first task was to increase the number of epochs sufficiently to get the model to converge, followed by adjusting the learning rate.
Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination but the ideal combination is data dependent.


## Generalisation

Generalisation is the model's ability to generalise well to untrained data.
Increasing the complexity to suit the training data results in over-fitting leading to the decrease in performance of the model when predicting the test data.
So it is essential to keep our model simple and keep in mind the assumptions governing generalisation namely i.i.d, stationary distribution and drawing examples from the same distribution.


## Training and test sets

This module highlights on the importance of slicing our data set into training and test data sets.
One should never train on the test data set as training on the test data over and over causes the model to overfit with the test data whereas it may not well when predicting an unseen data set.
The playground exercise helps us conclude that by reducing learning rate, test loss drops to a value much closer to training loss and that batch size does not cause a great influence in training as well as test loss.


## Validation set

Assessment of check your intuition:

![ML Validation set](https://user-images.githubusercontent.com/81472530/114302049-f3e4a100-9ae4-11eb-85bd-9df81d6736f1.jpg)

This module emphasises on creating an additional data set called validation set to validate the training data and further tune the hyperparameters.
Finally using the model well built on training and validation data to make predictions on the test data set.
This is a better workflow because it creates fewer exposures to the test set.
The programming exercise focuses on shuffling and dividing the single data set and checking the effectiveness of this workflow.


## Representation

### Feature engineering

This deals with converting raw data to a useful feature.
Taking the case of categorical features(which in their current form can't be much effective as they are in string form and not numerical values), the representation of one-hot encoding is discussed which comes handy in maintaining equal weightage for all the possibilities and including multiple indexes.
Qualities of good features, scaling the data, handling extreme outliers using logarithmic scaling or clipping feature values and the concept of binning are also covered.


## Feature crosses

Certain non-linearity in the model can be encoded by crossing two or more existing features to produce a synthetic feature(feature cross).
The new feature can also be added to the linear formula just like any other feature.
The playground exercise examined a non-linear output surface(elliptical in this case) which could be encoded using a feature cross but not using a linear model without crossing.
The programming exercise focused on crossing bins to create feature crosses.
Assessment of check your understanding:

![ML Feature crosses](https://user-images.githubusercontent.com/81472530/114303077-ca7a4400-9ae9-11eb-9764-1fa5bf14e319.jpg)


## Regularisation: Simplicity

The playground exercise depicts a case of over-crossing.
Regularisation refers to the penalising of the weights of features to avoid over-fitting.
Complexity is quantified using the L2 regularization formula, which defines the regularization term as the sum of the squares of all the feature weights.
This is added to the loss function after multiplication of lambda. Lambda is the factor accounting for the amount of regularisation effect.
Increasing lambda makes the weight distribution more like the Gaussian bell curve and lowering its value results in flatter distribution.
The playground exercise is based on altering the regularisation rate to minimise the difference between test and training loss.
Assessment of check your understanding:

![ML Regularisation](https://user-images.githubusercontent.com/81472530/114303461-debf4080-9aeb-11eb-9e70-461d14d37d65.jpg)

![ML Simplicity](https://user-images.githubusercontent.com/81472530/114303554-52614d80-9aec-11eb-88d3-9e9d8f5ea795.jpg)


## Logistic regression

It is used to create a probability value as the output.
A sigmoid function is used along with the linear formula to ensure the function produces a value from 0 to 1.
The loss function for logistic regression is log loss and regularisation is more important in logistic regression.
It is due to the asymptotic nature of the sigmoid that leads to the increasing complexity of the weights which needs to be kept under check by early stopping or L2 regularisation.


## Classification

This module is an extension of logistic regresssion to classification tasks by converting the probability to a binary value using a classification threshold.
We can summarize the model using a 2x2 matrix containg the four possible outcomes namely true positive(TP), false positive(FP), true negative(TN), false negative(FN).
Then evaluate classification models using metrics(accuracy, precision and recall) derived from these four outcomes. Accuracy is the fraction of correct predictions.
Accuracy fails to do a good job in class-imbalanced sets. 
Precision is the fraction of correct positive predictions. Recall is the fraction of positives identified correctly.
Precision and recall often act in inverse relation i.e increasing one reduces the other.
Assessment of check your understanding:

![ML Precision and recall](https://user-images.githubusercontent.com/81472530/114304706-49737a80-9af2-11eb-8bfa-a446b336902e.jpg)

ROC shows the performance of the classification by plotting TPR and FPR. AUC is the area under ROC curve.
AUC can be interpreted as the probability of correctly predicting a pair of positive and negative picked randomly from the set.
It does not depend on scaling or classification threshold.

![ML ROC](https://user-images.githubusercontent.com/81472530/114307785-b2f98600-9afe-11eb-88cd-4b3ebbe97510.jpg)

![ML AUC](https://user-images.githubusercontent.com/81472530/114307823-e6d4ab80-9afe-11eb-8552-6c39aa51db7f.jpg)

Prediction bias is a quantity that measures how far apart the average of predictions and observations are.
Though a significant prediction bias implies that the model is not good, low value of prediction bias does not mean that the model is perfect.


## Regularisation: Sparsity

In large data sets, for faster and more efficient processing, it is necessary that some weights are 0(which does not happen in L2).
This is attained by using L1 regularisation, which penalises |weight|.
The playground exercise shows an example where changing from L2 to L1 reduces the value of weights and decreases the test loss.
Assessment of check your understanding:

![ML sparsity](https://user-images.githubusercontent.com/81472530/114308588-23ee6d00-9b02-11eb-88d9-519bcc012afa.jpg)



















