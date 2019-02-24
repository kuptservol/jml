package ru.kuptservol.jml.layer;

import java.util.Arrays;
import java.util.Optional;

import lombok.Builder;
import ru.kuptservol.jml.activation.function.ActivationFunction;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.optimization.Regularization;
import ru.kuptservol.jml.weight.initializer.WeightInitializer;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
public class ConvolutionalLayer {

    @Builder.Default
    private double learningRate = 0.1;
    @Builder.Default
    private double momentumCoeff = 0;
    @Builder.Default
    private WeightInitializer weightInitializer = WeightInitializers.GAUSSIAN(1);
    @Builder.Default
    private ActivationFunction activationFunction = ActivationFunctions.SIGMOID;

    private final int in;
    private final int out;

    private final int filtersNum;
    private final int inputFeatureMapsNum;
    private final int filterHeight;
    private final int filterWidth;

    private final int imageX;
    private final int imageY;

    private final double[][][] sharedWeights;
    private final double[][][] featureMaps;
    private final double[][][] featureMapsZ;
    private final double[] sharedBiases;

    private double[][] dwWithMomentum;
    private double[][] prevLayerActivations;
    private double[][][] deltaSharedWeights;
    private double[] deltaSharedBiases;
    private double[] z;

    @Builder
    public ConvolutionalLayer(
            int in,
            int out,
            Double learningRate,
            WeightInitializer weightInitializer,
            ActivationFunction activationFunction,
            Double momentumCoeff,
            int filtersNum,
            int inputFeatureMapsNum,
            int filterHeight,
            int filterWidth,
            int imageX,
            int imageY)
    {
        this.in = in;
        this.out = out;
        this.weightInitializer = Optional.ofNullable(weightInitializer).orElse(this.weightInitializer);
        this.filtersNum = filtersNum;
        this.inputFeatureMapsNum = inputFeatureMapsNum;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        this.imageX = imageX;
        this.imageY = imageY;

        this.sharedWeights = new double[filtersNum][filterHeight][filterWidth];
        for (int i = 0; i < filtersNum; i++) {
            sharedWeights[i] = this.weightInitializer.initWeights(filterHeight, filterWidth);
        }
        this.deltaSharedWeights = new double[filtersNum][filterHeight][filterWidth];

        this.sharedBiases = this.weightInitializer.initBiases(filtersNum);
        this.deltaSharedBiases = new double[filtersNum];

        this.featureMaps = new double[filtersNum][imageX - filterHeight + 1][imageY - filterWidth + 1];

        this.dwWithMomentum = new double[in][out];
        this.learningRate = Optional.ofNullable(learningRate).orElse(this.learningRate);
        this.activationFunction = Optional.ofNullable(activationFunction).orElse(this.activationFunction);

        this.z = new double[out];

        this.featureMapsZ = new double[filtersNum][imageX - filterHeight + 1][imageY - filterWidth + 1];

        this.prevLayerActivations = new double[imageX][imageY];
        this.momentumCoeff = Optional.ofNullable(momentumCoeff).orElse(this.momentumCoeff);
    }

    public double[] forward(double[][] inputActivations) {
        prevLayerActivations = inputActivations;

        /* z = Wx + b
         * a = actv(z)
         */
        for (int filter = 0; filter < filtersNum; filter++) {
            for (int fX = 0; fX < imageX - filterWidth + 1; fX++) {
                for (int fY = 0; fY < imageY - filterHeight + 1; fY++) {
                    double[][] filterSharedWeight = sharedWeights[filter];
                    double sharedFilterBias = sharedBiases[filter];

                    featureMapsZ[filter][fX][fY] =
                            M.convR(inputActivations, fX + filterWidth, fY + filterHeight, filterSharedWeight) + sharedFilterBias;

                    featureMaps[filter][fX][fY] = activationFunction.activate(new double[]{featureMapsZ[filter][fX][fY]})[0];
                }
            }
        }

        return activationFunction.activate(z);
    }

//    /**
//     * @param dCostDaWNextLayer = W*dC/dA(l+1)
//     */
//    public double[] backprop(double[] dCostDaWNextLayer) {
//        /* dC/da */
//        double[] dCDa = M.hadamartR(dCostDaWNextLayer, activationFunction.dADz(z));
//        /* dC/db */
//        double[] dCDb = dCDa;
//        /* dC/dw */
//        double[][] dCDw = M.dotR(prevLayerActivations, dCDa);
//
//        M.plus(deltaSharedBiases, dCDb);
//        M.plus(deltaSharedWeights, dCDw);
//
//        return M.dotR(featureMaps, dCDa);
//    }
//
//    @Override
//    public double[] lastLayerBackprop(double[] dCostDa) {
//        return dCostDa;
//    }
//
//    @Override
//    public void onBatchStarted() {
//        resetBatchWeights();
//    }
//
//    @Override
//    public void setLearningRate(double learningRate) {
//        this.learningRate = learningRate;
//    }
//
//    private void resetBatchWeights() {
//        this.deltaSharedWeights = new double[filtersNum][filterHeight][filterWidth];
//        deltaSharedBiases = new double[filtersNum];
//    }
//
//    @Override
//    public double norm(Regularization reg) {
//        return Arrays.stream(sharedWeights).map(reg::norm).reduce((a, b) -> a + b).get();
//    }
//
//    @Override
//    public void onBatchFinished(int batchSize, Regularization reg) {
//        updateBatchWeights(batchSize, reg);
//    }
//
//    private void updateBatchWeights(int batchSize, Regularization reg) {
//        M.F(dwWithMomentum, deltaSharedWeights, (momentum, dw) -> momentum * momentumCoeff - (1.0 / batchSize) * dw * learningRate);
//
//        M.F(featureMaps, dwWithMomentum, (w, dwWithMomentum) -> w - reg.reg(learningRate, batchSize, w) + dwWithMomentum);
//        M.F(sharedBiases, deltaSharedBiases, (b, db) -> b - db * (learningRate / batchSize));
//    }
//
//    @Override
//    public double[] dActDZ() {
//        return activationFunction.dADz(z);
//    }
}
