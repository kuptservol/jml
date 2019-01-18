package ru.kuptservol.jml.layer;

import java.io.Serializable;

import ru.kuptservol.jml.optimization.Regularization;

/**
 * @author Sergey Kuptsov
 */
public interface Layer extends Serializable {

    double[] forward(double[] prevActivations);

    double[] backprop(double[] dCostDaNextLayer);

    double[] lastLayerBackprop(double[] dCostDaNextLayer);

    void onBatchStarted();

    void setLearningRate(double learningRate);

    double norm(Regularization regularization);

    void onBatchFinished(int batchSize, Regularization l2Reg);

    double[] dActDZ();

}
