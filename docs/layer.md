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
**Forward:**
 
<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{j}}^{l}&space;=&space;activationfunc&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{j}}^{l}&space;=&space;activationfunc&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" title="\textup{a}{_{j}}^{l} = activationfunc (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{j}}^{l}&space;=&space;activationfunc&space;({z}{_{j}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{j}}^{l}&space;=&space;activationfunc&space;({z}{_{j}}^{l})" title="\textup{a}{_{j}}^{l} = activationfunc ({z}{_{j}}^{l})" /></a>

If activation function is sigmoid then

 <a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" title="\textup{a}{_{j}}^{l} = \sigma (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l})" /></a>

**Derivative:**

**By W**:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;\frac{\partial&space;activationfunc}{\partial&space;z}&space;*&space;a_{l-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;\frac{\partial&space;activationfunc}{\partial&space;z}&space;*&space;a_{l-1}" title="\frac{\partial CostF}{\partial w_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* \frac{\partial activationfunc}{\partial z} * a_{l-1}" /></a>

If activation function is sigmoid then

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" title="\frac{\partial CostF}{\partial w_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))' * a_{l-1}" /></a>

**By b**:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;\frac{\partial&space;activationfunc}{\partial&space;z}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;\frac{\partial&space;activationfunc}{\partial&space;z}" title="\frac{\partial CostF}{\partial w_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* \frac{\partial activationfunc}{\partial z}" /></a>

If activation function is sigmoid then

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
        double[] dCDa = M.hadamartR(dCostDaWNextLayer, activationFunction.dADz(z));
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
