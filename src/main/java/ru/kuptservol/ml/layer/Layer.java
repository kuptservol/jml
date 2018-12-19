package ru.kuptservol.ml.layer;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface Layer extends Serializable {

    double[] forward(double[] prevActivations);

    double[] backprop(double[] dCostDaNextLayer);

    void onBatchStarted();

    void onBatchFinished(int batchSize);
}
