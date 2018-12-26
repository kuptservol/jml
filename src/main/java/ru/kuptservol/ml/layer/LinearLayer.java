package ru.kuptservol.ml.layer;

import java.util.Optional;

import ru.kuptservol.ml.activation.function.ActivationFunction;
import ru.kuptservol.ml.activation.function.ActivationFunctions;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.weight.initializer.WeightInitializer;
import ru.kuptservol.ml.weight.initializer.WeightInitializers;
import lombok.Builder;

/**
 * @author Sergey Kuptsov
 */
public class LinearLayer implements Layer {

    @Builder.Default
    private double learningRate = 0.1;
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
    public LinearLayer(int in, int out, Double learningRate, WeightInitializer weightInitializer, ActivationFunction activationFunction) {
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
        double[] dCDb = dCDa;
        /* dC/dw */
        double[][] dCDw = M.dotR(prevLayerActivations, dCDa);

        M.plus(deltaBiases, dCDb);
        M.plus(deltaWeights, dCDw);

        return M.dotR(weights, dCDa);
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
}
