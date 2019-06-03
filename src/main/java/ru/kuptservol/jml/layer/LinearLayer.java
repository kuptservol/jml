package ru.kuptservol.jml.layer;

import java.util.Optional;

import lombok.Builder;
import ru.kuptservol.jml.activation.function.ActivationFunction;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.optimization.Dropout;
import ru.kuptservol.jml.optimization.Optimizations;
import ru.kuptservol.jml.optimization.Optimizer;
import ru.kuptservol.jml.optimization.Regularization;
import ru.kuptservol.jml.weight.initializer.WeightInitializer;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
public class LinearLayer implements Layer {

    @Builder.Default
    // [nu:]
    private double learningRate = 0.1;
    @Builder.Default
    // Beta
    private double momentumCoeff = 0;
    @Builder.Default
    private Dropout dropout = Optimizations.Dropout(0);
    @Builder.Default
    private WeightInitializer weightInitializer = WeightInitializers.Gaussian(1);
    @Builder.Default
    private ActivationFunction activationFunction = ActivationFunctions.Sigmoid;

    private final int in;
    private final int out;

    private double[][] weights;
    // for prev in momentum
    private double[][] weightBatchGradsPrev;
    private double[] prevLayerActivations;
    private double[][] weightBatchGrads;
    private double[] biasBatchGrads;
    private double[] biasBatchGradsPrev;
    private double[] biases;
    private double[] z;

    private Optimizer optimizer;

    @Builder
    public LinearLayer(
            int in,
            int out,
            Double learningRate,
            Double dropout,
            WeightInitializer weightInitializer,
            ActivationFunction activationFunction,
            Double momentumCoeff,
            Optimizer optimizer)
    {
        this.in = in;
        this.out = out;

        this.weightInitializer = Optional.ofNullable(weightInitializer).orElse(this.weightInitializer);

        this.weights = this.weightInitializer.initWeights(in, out);
        this.weightBatchGradsPrev = new double[in][out];
        this.weightBatchGrads = new double[in][out];

        this.biases = this.weightInitializer.initBiases(out);
        this.biasBatchGradsPrev = new double[out];
        this.biasBatchGrads = new double[out];

        this.prevLayerActivations = new double[in];
        this.z = new double[out];
        this.learningRate = Optional.ofNullable(learningRate).orElse(this.learningRate);
        this.momentumCoeff = Optional.ofNullable(momentumCoeff).orElse(this.momentumCoeff);
        this.activationFunction = Optional.ofNullable(activationFunction).orElse(this.activationFunction);
        this.dropout = Optional.ofNullable(dropout).map(Optimizations::Dropout).orElse(this.dropout);

        this.optimizer = optimizer;
    }

    @Override
    public double[] forward(double[] inputActivations) {
        // todo: if dropout and test - multiply weights 1/(dropout probability) times each activation
        prevLayerActivations = inputActivations;

        /* z = Wx + b
         * a = actv(z)
         */
        z = M.plusR(
                M.dotR(inputActivations, weights),
                biases
        );

        double[] a = activationFunction.activate(z);

        return M.hadamartR(a, dropout.mask());
    }

    /**
     * @param dCostDaWNextLayer = W*dC/dA(l+1)
     */
    @Override
    public double[] backprop(double[] dCostDaWNextLayer) {
        /* dC/da */
        double[] dCDa = M.hadamartR(dCostDaWNextLayer, activationFunction.dADz(z));
        dCDa = M.hadamartR(dCDa, dropout.mask());
        /* dC/db */
        double[] dCDb = dCDa;
        /* dC/dw */
        double[][] dCDw = M.dotR(prevLayerActivations, dCDa);

        M.plus(biasBatchGrads, dCDb);
        M.plus(weightBatchGrads, dCDw);

        return M.dotR(weights, dCDa);
    }

    @Override
    public double[] lastLayerBackprop(double[] dCostDa) {
        /* dC/db */
        double[] dCDb = dCostDa;
        /* dC/dw */
        double[][] dCDw = M.dotR(prevLayerActivations, dCostDa);

        M.plus(biasBatchGrads, dCDb);
        M.plus(weightBatchGrads, dCDw);

        return M.dotR(weights, dCostDa);
    }

    @Override
    public void onBatchStarted() {
        initDropoutMask();
        resetBatchWeights();
    }

    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    private void initDropoutMask() {
        dropout.initMask(out);
    }

    private void resetBatchWeights() {
        weightBatchGrads = new double[in][out];
        biasBatchGrads = new double[out];
    }

    @Override
    public double norm(Regularization reg) {
        return reg.norm(weights);
    }

    @Override
    public void onBatchFinished(int batchSize, Regularization reg) {
        updateBatchWeights(batchSize, reg);
    }

    private void updateBatchWeights(int batchSize, Regularization reg) {
        // calculate weight grads with optimizer
        optimizer.optimize(weightBatchGrads);

        M.F(weights, weightBatchGrads, (weights, weightBatchGrad)
                -> weights - (1.0 / batchSize) * weightBatchGrad * learningRate - reg.reg(learningRate, batchSize, weights));

        optimizer.optimize(biasBatchGrads);

        // calculate new biases
        M.F(biases, biasBatchGrads,
                (bias, biasBatchGrad) -> bias - (1.0 / batchSize) * biasBatchGrad * learningRate);
    }

    @Override
    public double[] dActDZ() {
        return activationFunction.dADz(z);
    }
}
