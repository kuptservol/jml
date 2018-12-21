# jml
### Yet Another Java ML Lib

Java ML library made for educational purposes mostly(no optimization, threads, gpu, etc) to study and practise ml theory things for people with strong java background - for those who find it difficult to study ml concepts(f.e. lin algebra) and python concepts(f.e. numpy) simultaneously - like me.

Made with pure java math without any libs except lombok, slf4j and junit, code organization is more common for java project with builders, interfaces, hierarchy and so on - so can be easily extended

#### MNIST
MNIST - is a database of hand-written digits. With 10 simple classes in java we can learn to determine digit on photo with 90% acccuracy within 15 minutes learning on one CPU.
Kaggle competiton https://www.kaggle.com/c/digit-recognizer/leaderboard

#### Content
* Math
* Train
* Weight initialization
* Loss functions
* Layer
* Activation Function
* Training
* Optimization

#### Math
Let's have layer function - we transform input from prev layer to output like this:

<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" title="\textup{a}{_{j}}^{l} = \sigma (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l})" /></a>

we try to minimize cost function.

<a href="https://www.codecogs.com/eqnedit.php?latex=CostF&space;=&space;\sum_x&space;(a&space;-&space;y(x))&space;=&space;CostF&space;-&space;\bigtriangleup&space;CostF&space;\rightarrow&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?CostF&space;=&space;\sum_x&space;(a&space;-&space;y(x))&space;=&space;CostF&space;-&space;\bigtriangleup&space;CostF&space;\rightarrow&space;0" title="CostF = \sum_x (a - y(x)) = CostF - \bigtriangleup CostF \rightarrow 0" /></a>

we want 
<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;CostF&space;<&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;CostF&space;<&space;0" title="\bigtriangleup CostF < 0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;CostF&space;=&space;\bigtriangleup&space;x&space;\frac{\partial&space;CostF{}}{\partial&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;CostF&space;=&space;\bigtriangleup&space;x&space;\frac{\partial&space;CostF{}}{\partial&space;x}" title="\bigtriangleup CostF = \bigtriangleup x \frac{\partial CostF{}}{\partial x}" /></a>

if we make <a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;x&space;=&space;-\eta&space;\frac{\partial&space;CostF{}}{\partial&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;x&space;=&space;-\eta&space;\frac{\partial&space;CostF{}}{\partial&space;x}" title="\bigtriangleup x = -\eta \frac{\partial CostF{}}{\partial x}" /></a>

then 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;CostF&space;=&space;-&space;\bigtriangleup&space;x&space;(\frac{\partial&space;CostF{}}{\partial&space;x})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;CostF&space;=&space;-&space;\bigtriangleup&space;x&space;(\frac{\partial&space;CostF{}}{\partial&space;x})^2" title="\bigtriangleup CostF = - \bigtriangleup x (\frac{\partial CostF{}}{\partial x})^2" /></a>

less than 0 by design

so if we know partial derivatives of w and b on each layer - we can update them and expect loss decreases 

<a href="https://www.codecogs.com/eqnedit.php?latex=w_k&space;\rightarrow&space;w_k{}'&space;=&space;w_k&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;w_k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_k&space;\rightarrow&space;w_k{}'&space;=&space;w_k&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;w_k}" title="w_k \rightarrow w_k{}' = w_k - \eta \bigtriangledown\frac{\partial CostF}{\partial w_k}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=b_l&space;\rightarrow&space;b_l{}'&space;=&space;b_l&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;b_l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_l&space;\rightarrow&space;b_l{}'&space;=&space;b_l&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;b_l}" title="b_l \rightarrow b_l{}' = b_l - \eta \bigtriangledown\frac{\partial CostF}{\partial b_l}" /></a>

To know derivatives we apply chain rule - cause result of cost function is

<a href="https://www.codecogs.com/eqnedit.php?latex=Cost&space;=&space;CostF(a_j(&space;{a_{j-1}}...)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Cost&space;=&space;CostF(a_j(&space;{a_{j-1}}...)))" title="Cost = CostF(a_j( {a_{j-1}}...)))" /></a>

So derivative is  

<a href="https://www.codecogs.com/eqnedit.php?latex={\frac{\partial&space;Cost}{\partial&space;x}}&space;=&space;{\frac{\partial&space;Cost}{\partial&space;a_j}}&space;{\frac{\partial&space;a_j}{\partial&space;a_{j-1}}}&space;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\frac{\partial&space;Cost}{\partial&space;x}}&space;=&space;{\frac{\partial&space;Cost}{\partial&space;a_j}}&space;{\frac{\partial&space;a_j}{\partial&space;a_{j-1}}}&space;..." title="{\frac{\partial Cost}{\partial x}} = {\frac{\partial Cost}{\partial a_j}} {\frac{\partial a_j}{\partial a_{j-1}}} ..." /></a>

Remember that  <a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{}}^{l}&space;=&space;\sigma&space;(\sum&space;\textup{w}{_{}}^{l}&space;\textup{a}{_{}}^{l-1}&space;&plus;&space;\textup{b}{_{}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{}}^{l}&space;=&space;\sigma&space;(\sum&space;\textup{w}{_{}}^{l}&space;\textup{a}{_{}}^{l-1}&space;&plus;&space;\textup{b}{_{}}^{l})" title="\textup{a}{_{}}^{l} = \sigma (\sum \textup{w}{_{}}^{l} \textup{a}{_{}}^{l-1} + \textup{b}{_{}}^{l})" /></a>

And <a href="https://www.codecogs.com/eqnedit.php?latex=CostF&space;=&space;\frac{1}{n}\sum&space;(&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?CostF&space;=&space;\frac{1}{n}\sum&space;(&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})&space;-&space;y)^2" title="CostF = \frac{1}{n}\sum ( \sigma (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l}) - y)^2" /></a>

If const-function = MSE, then for last linear layer:

<a href="https://www.codecogs.com/eqnedit.php?latex=CostF'&space;=&space;{((a_l-y)^2)}'*{(w_la_{l-1}&plus;b_l)}'*(\sigma(w_la_{l-1}&plus;b_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?CostF'&space;=&space;{((a_l-y)^2)}'*{(w_la_{l-1}&plus;b_l)}'*(\sigma(w_la_{l-1}&plus;b_l))'" title="CostF' = {((a_l-y)^2)}'*{(w_la_{l-1}+b_l)}'*(\sigma(w_la_{l-1}+b_l))'" /></a>

By w:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_l}&space;=&space;(a_l-y)*a_{l-1}*(\sigma(w_la_{l-1}&plus;b_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_l}&space;=&space;(a_l-y)*a_{l-1}*(\sigma(w_la_{l-1}&plus;b_l))'" title="\frac{\partial CostF}{\partial w_l} = (a_l-y)*a_{l-1}*(\sigma(w_la_{l-1}+b_l))'" /></a>

By b:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;b_l}&space;=&space;(a_l-y)*(\sigma(w_la_{l-1}&plus;b_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;b_l}&space;=&space;(a_l-y)*(\sigma(w_la_{l-1}&plus;b_l))'" title="\frac{\partial CostF}{\partial b_l} = (a_l-y)*(\sigma(w_la_{l-1}+b_l))'" /></a>

Then going back from last layer to first

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;a_{l&plus;1}}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*\frac{\partial&space;(w_{l&plus;1}\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))&plus;b_{l&plus;1}))}{\partial&space;a_l}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;(w_{l&plus;1}z_l&plus;b_{l&plus;1})}{\partial&space;a_{l}}*&space;(\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))'&space;*&space;(w_l*a_{l-1}&space;&plus;&space;b_l)'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;a_{l&plus;1}}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*\frac{\partial&space;(w_{l&plus;1}\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))&plus;b_{l&plus;1}))}{\partial&space;a_l}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;(w_{l&plus;1}z_l&plus;b_{l&plus;1})}{\partial&space;a_{l}}*&space;(\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))'&space;*&space;(w_l*a_{l-1}&space;&plus;&space;b_l)'" title="\frac{\partial CostF}{\partial a_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * \frac{\partial a_{l+1}}{\partial a_{l}} = \frac{\partial CostF}{\partial a_{l+1}} *\frac{\partial (w_{l+1}\sigma(w_l*a_{l-1} + b_l))+b_{l+1}))}{\partial a_l} = \frac{\partial CostF}{\partial a_{l+1}} * \frac{\partial (w_{l+1}z_l+b_{l+1})}{\partial a_{l}}* (\sigma(w_l*a_{l-1} + b_l))' * (w_l*a_{l-1} + b_l)'" /></a>

By w:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" title="\frac{\partial CostF}{\partial w_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))' * a_{l-1}" /></a>

By b:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;b_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;b_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'" title="\frac{\partial CostF}{\partial b_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))'" /></a>

So for each layer we must know <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}" title="\frac{\partial CostF}{\partial a_{l+1}}" /></a> from next layer,
current activations, weights from next layer and activations from prev layer - for first layer this is inputs in model

##### Train
```java
public interface Trainer extends Serializable {
    void train(Model m, DataSet dataSet);
}
```
###### SGD
In stochastic gradient descent we don't want to wait all inputs proceed to make one update of w - instead we average derivatives from some small number of inputs - update weights and take another portion. When we proceed all portions - it's called epoch.
```java
    @Override
    public void train(Model m, DataSet dataSet) {
        double[][] trainX = dataSet.train.x;
        double[][] trainY = dataSet.train.y;
        for (int i = 0; i < epochs; i++) {
            m.trainListener.onEpochStarted(i);
            M.shuffle(trainX, trainY);
            M.Data[] miniBatches = M.chunk(trainX, trainY, batchSize);
            for (int j = 0; j < miniBatches.length; j++) {
                M.Data miniBatch = miniBatches[j];
                trainOnMiniBatch(miniBatch, m, j);
            }
        }
    }

    private void trainOnMiniBatch(M.Data trainMiniBatch, Model m, int batchId) {
        m.layers.forEach(Layer::onBatchStarted);
        for (int i = 0; i < trainMiniBatch.size; i++) {
            double[] activations = m.layers.forward(trainMiniBatch.x[i]);
            m.layers.backprop(m.costFunction.backprop(activations, trainMiniBatch.y[i]));
        }
        m.layers.forEach(layer -> layer.onBatchFinished(trainMiniBatch.size));
    } 
```
##### Weight Initializers
```java
public interface WeightInitializer extends Serializable {
    double[][] init(int x, int y);
}
```
###### Gaussian
```java
    private final double limit;
    private final Random random = new Random();
    
    @Override
    public double[][] init(int x, int y) {
        double[][] vals = new double[x][y];
        return M.FR(v -> limit * random.nextGaussian(), vals);
    }
```
##### Loss functions
```java
public interface CostFunction extends Serializable {
    MetricsResult execute(Model m, double[][] trainX, double[][] trainY);
    double[] backprop(double[] activations, double[] y);
}
```
###### Mean Square Error
Forward: <a href="https://www.codecogs.com/eqnedit.php?latex=C(w,b)=\frac{1}n{}&space;\sum(y(x)&space;-&space;a)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C(w,b)=\frac{1}n{}&space;\sum(y(x)&space;-&space;a)^2" title="C(w,b)=\frac{1}n{} \sum(y(x) - a)^2" /></a>

Derivative: <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;C}{\partial&space;a}&space;=&space;y(x)-a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;C}{\partial&space;a}&space;=&space;y(x)-a" title="\frac{\partial C}{\partial a} = y(x)-a" /></a>
```java
public class MSE implements CostFunction {
    @Override
    public MetricsResult execute(Model m, double[][] X, double[][] Y) {
        double cost = 0;
        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.evaluate(X[i]) - m.resultFunction.apply(Y[i]), 2) / X.length;
        }
        return new SimpleMetricsResult(cost, "MSE: %.3f");
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
 ``` 
 ##### Layer
 ```java
public interface Layer extends Serializable {
    double[] forward(double[] prevActivations);
    double[] backprop(double[] dCostDaNextLayer);
    void onBatchStarted();
    void onBatchFinished(int batchSize);
}
 ```
 ###### Linear
Forward: <a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" title="\textup{a}{_{j}}^{l} = \sigma (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l})" /></a>

 
Derivative:

By w:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" title="\frac{\partial CostF}{\partial w_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))' * a_{l-1}" /></a>

By b:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;b_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;b_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'" title="\frac{\partial CostF}{\partial b_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))'" /></a>


 ```java
    @Override
    public double[] forward(double[] inputActivations) {
        prevLayerActivations = inputActivations;

        /* z = Wx + b
         * a = actv(z)
         */
        z = M.plusR(
                M.dotR(inputActivations, weights),
                biases
        );

        return activationFunction.activate(z);
    }

    /**
     * @param dCostDaWNextLayer = W(l+1)*dC/dA(l+1)
     */
    @Override
    public double[] backprop(double[] dCostDaWNextLayer) {
        /* dC/da */
        double[] dCostDa = M.hadamartR(dCostDaWNextLayer, activationFunction.dADz(z));
        /* dC/db */
        double[] dCDb = dCostDa;
        /* dC/dw */
        double[][] dCDw = M.dotR(prevLayerActivations, dCostDa);

        M.plus(deltaBiases, dCDb);
        M.plus(deltaWeights, dCDw);

        return M.dotR(weights, dCostDa);
    }

    @Override
    public void onBatchStarted() {
        deltaWeights = new double[in][out];
        deltaBiases = new double[out];
    }

    @Override
    public void onBatchFinished(int batchSize) {
        M.F(weights, deltaWeights, (w, dw) -> w - dw * (learningRate / batchSize));
        M.F(biases, deltaBiases, (b, db) -> b - db * (learningRate / batchSize));
    }
 ```
##### Activation Function
```java
public interface ActivationFunction extends Serializable {
    double[] activate(double[] values);
    double[] dADz(double[] z);
}
```
###### Sigmoid
```java
    @Override
    public double[] activate(double[] z) {
        return M.FR(this::sigmoid, z);
    }
    @Override
    public double[] dADz(double[] z) {
        return M.FR(v -> sigmoid(v) * (1 - sigmoid(v)), z);
    }
    private double sigmoid(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }
```

  
##### With default settings:
From `MNISTest.learnWithDefaultSettings`
```java
DataSet mnist = DataSets.MNIST(Paths.get("/tmp/mnist"));

Model model = Models.linear(784, 30, 10)
    .resultFunction(ResultFunctions.MAX_INDEX)
    .metrics(Metrics.ACCURACY)
    .build();

model.train(mnist);

Path path = Paths.get("/tmp/mnist_model");

model.save(path);
Model modelLoaded = Models.load(path);
        
logger.debug("X1: " + M.asPixels(M.to(mnist.train.x[1], 28, 28)));
logger.debug("Expected answer: " + modelLoaded.resultFunction.apply(mnist.train.y[1]));
logger.debug("Model answer: " + modelLoaded.evaluate(mnist.train.x[1]));
```
Output
```bash
2018-12-19 14:12:05:926 +0300 [main] INFO Epoch 0 train accuracy 26,893 % test accuracy 26,810 % MSE: 12,620
...
2018-12-19 14:41:55:975 +0300 [main] INFO Epoch 88 train accuracy 88,543 % test accuracy 88,940 % MSE: 1,954

2018-12-19 14:47:27:645 +0300 [main] DEBUG X1:                             
                                       
               *****        
              ******        
             *********      
           ***********      
           ***********      
          ************      
         *********  ***     
        ******      ***     
       *******      ***     
       ****         ***     
       ***          ***     
      ****          ***     
      ****        *****     
      ***        *****      
      ***       ****        
      ***      ****         
      *************         
      ***********           
      *********             
       *******              
                            
2018-12-19 14:47:27:646 +0300 [main] DEBUG Expected answer: 0.0
2018-12-19 14:47:27:662 +0300 [main] DEBUG Model answer: 0.0

```


