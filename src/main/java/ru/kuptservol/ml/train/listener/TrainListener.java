package ru.kuptservol.ml.train.listener;

import java.io.Serializable;
import java.util.Optional;

import ru.kuptservol.ml.metric.result.Metric;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface TrainListener extends Serializable {
    void onEpochStarted(int epochId);

    void onEpochFinished(int epochId, Optional<Metric> trainMetrics, Optional<Metric> testMetrics, Metric costMetrics);

    void onBatchStarted(int batchId);

    void onBatchFinished(int batchId);

    void onTrainStarted();

    void onTrainFinished(Model m);
}
