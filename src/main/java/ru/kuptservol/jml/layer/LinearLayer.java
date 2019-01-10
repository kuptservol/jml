package ru.kuptservol.jml.layer;

import java.util.Optional;

import lombok.Builder;
import ru.kuptservol.jml.activation.function.ActivationFunction;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.optimization.Dropout;
import ru.kuptservol.jml.optimization.Optimizations;
import ru.kuptservol.jml.optimization.Regularization;
import ru.kuptservol.jml.weight.initializer.WeightInitializer;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
public class LinearLayer implements Layer {

    @Builder.Default
    private double learningRate = 0.1;
    @Builder.Default
    private Dropout dropout = Optimizations.DROPOUT(1);
    @Builder.Default
    private WeightInitializer weightInitializer = WeightInitializers.gaussian(1);
    @Builder.Default
    private ActivationFunction activationFunction = ActivationFunctions.SIGMOID;

    private final int in;
    private final int out;

    private double[][] weights;
    private double[] prevLayerActivations;
    private double[][] deltaWeights;
    private double[] deltaBiases;
    private double[] biases;
    private double[] z;

    @Builder
    public LinearLayer(int in, int out, Double learningRate, Double dropout, WeightInitializer weightInitializer, ActivationFunction activationFunction) {
        this.in = in;
        this.out = out;
        this.weightInitializer = Optional.ofNullable(weightInitializer).orElse(this.weightInitializer);
        this.weights = this.weightInitializer.init(in, out);
        this.deltaWeights = new double[in][out];
        this.biases = this.weightInitializer.init(out);
        this.deltaBiases = new double[out];
        this.prevLayerActivations = new double[in];
        this.z = new double[out];
        this.learningRate = Optional.ofNullable(learningRate).orElse(this.learningRate);
        this.activationFunction = Optional.ofNullable(activationFunction).orElse(this.activationFunction);
        this.dropout = Optional.ofNullable(dropout).map(Optimizations::DROPOUT).orElse(this.dropout);
    }

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

        M.plus(deltaBiases, dCDb);
        M.plus(deltaWeights, dCDw);

        return M.dotR(weights, dCDa);
    }

    @Override
    public double[] lastLayerBackprop(double[] dCostDa) {
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
        initDropoutMask();
        resetBatchWeights();
    }

    private void initDropoutMask() {
        dropout.initMask(out);
    }

    private void resetBatchWeights() {
        deltaWeights = new double[in][out];
        deltaBiases = new double[out];
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
        M.F(weights, deltaWeights, (w, dw) -> w - reg.reg(learningRate, batchSize, w) - dw * (learningRate / batchSize));
        M.F(biases, deltaBiases, (b, db) -> b - db * (learningRate / batchSize));
    }

    @Override
    public double[] dActDZ() {
        return activationFunction.dADz(z);
    }
}
