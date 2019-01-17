#### Optimization strategies
##### L2 regularization
//todo: reword
The idea of L2 regularization is to add an extra term to the cost function, a term called the regularization term(lambda)

<a href="https://www.codecogs.com/eqnedit.php?latex=C&space;=&space;C_0&space;&plus;&space;\frac{\lambda&space;}{2n}\sum_w&space;w^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C&space;=&space;C_0&space;&plus;&space;\frac{\lambda&space;}{2n}\sum_w&space;w^{2}" title="C = C + \frac{\lambda }{2n}\sum_w w^{2}" /></a>

and 

<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;\rightarrow&space;w'&space;=&space;w\left(1&space;-&space;\frac{\eta&space;\lambda}{n}&space;\right)&space;-&space;\eta&space;\frac{\partial&space;C_0}{\partial&space;w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;\rightarrow&space;w'&space;=&space;w\left(1&space;-&space;\frac{\eta&space;\lambda}{n}&space;\right)&space;-&space;\eta&space;\frac{\partial&space;C_0}{\partial&space;w}" title="w \rightarrow w' = w\left(1 - \frac{\eta \lambda}{n} \right) - \eta \frac{\partial C_0}{\partial w}" /></a>

//todo: reword
Large weights will only be allowed if they considerably improve the first part of the cost function.
The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. 
By contrast, a network with large weights may change its behaviour quite a bit in response to small changes in the input. 
Regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, 
and are resistant to learning peculiarities of the noise in the training data.
##### L1 regularization
<a href="https://www.codecogs.com/eqnedit.php?latex=C&space;=&space;C_0&space;&plus;&space;\frac{\lambda}{n}&space;\sum_w&space;|w|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C&space;=&space;C_0&space;&plus;&space;\frac{\lambda}{n}&space;\sum_w&space;|w|" title="C = C_0 + \frac{\lambda}{n} \sum_w |w|" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;C}{\partial&space;w}&space;=&space;\frac{\partial&space;C_0}{\partial&space;w}&space;&plus;&space;\frac{\lambda}{n}&space;\,&space;{\rm&space;sgn}(w)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;C}{\partial&space;w}&space;=&space;\frac{\partial&space;C_0}{\partial&space;w}&space;&plus;&space;\frac{\lambda}{n}&space;\,&space;{\rm&space;sgn}(w)" title="\frac{\partial C}{\partial w} = \frac{\partial C_0}{\partial w} + \frac{\lambda}{n} \, {\rm sgn}(w)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;\rightarrow&space;w'&space;=&space;w-\frac{\eta&space;\lambda}{n}&space;\mbox{sgn}(w)&space;-&space;\eta&space;\frac{\partial&space;C_0}{\partial&space;w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;\rightarrow&space;w'&space;=&space;w-\frac{\eta&space;\lambda}{n}&space;\mbox{sgn}(w)&space;-&space;\eta&space;\frac{\partial&space;C_0}{\partial&space;w}" title="w \rightarrow w' = w-\frac{\eta \lambda}{n} \mbox{sgn}(w) - \eta \frac{\partial C_0}{\partial w}" /></a>

In L1 regularization, the weights shrink by a constant amount toward 0. In L2 regularization, the weights shrink by an amount which is proportional to w. And so when a particular weight has a large magnitude, |w|, L1 regularization shrinks the weight much less than L2 regularization does. By contrast, when |w| is small, L1 regularization shrinks the weight much more than L2 regularization. The net result is that L1 regularization tends to concentrate the weight of the network in a relatively small number of high-importance connections, while the other weights are driven toward zero.


##### Early stopping to determine the number of training epochs
arly stopping means that at the end of each epoch we should compute the classification accuracy on the validation data. 
When that stops improving, terminate.
Furthermore, early stopping also automatically prevents us from overfitting.

##### Measuring accuracy on validation data sets
When we set the hyper-parameters based on evaluations of the test_data it's possible we'll end 
up overfitting our hyper-parameters to the test_data. We may end up finding hyper-parameters which 
fit particular peculiarities of the test_data, 
but where the performance of the network won't generalize to other data sets.
We guard against that by figuring out the hyper-parameters using the validation_data
##### Dropout
Dropout is a technic to delete randomly choosed parts of neurons on every mini-batch step.

When we dropout different sets of neurons, it's rather like we're training different neural networks. 
And so the dropout procedure is like averaging the effects of a very large number of different networks. 
The different networks will overfit in different ways, and so, hopefully, the net effect of dropout will be to reduce overfitting.

   
##### Increasing train data size
##### Hyper-parameters auto tunning
##### Learninig rate auto tunning

Start from some initial rather small value -> increase it and try until validation cost decreases.

##### Varying learning rate during training(adaptive learning rate)

It's likely that the weights are badly wrong. 
And so it's best to use a large learning rate that causes the weights to change quickly. 
Later, we can reduce the learning rate as we make more fine-tuned adjustments to our weights.

##### Epochs batch calculating parallellization
