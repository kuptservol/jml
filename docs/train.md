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
