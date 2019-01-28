#### Optimization strategies
##### Hyper-parameters auto tuning 
##### Learning rate auto tuning

Start from some initial rather small value -> increase it and try until validation cost decreases.

##### Varying learning rate during training(adaptive learning rate)

It's likely that the weights are badly wrong. 
And so it's best to use a large learning rate that causes the weights to change quickly. 
Later, we can reduce the learning rate as we make more fine-tuned adjustments to our weights.

##### Epochs batch calculating parallelization
##### SGD with momentum
