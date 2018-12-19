package ru.kuptservol.ml.train.listener;

import java.io.Serializable;
import java.util.Optional;

import ru.kuptservol.ml.metric.MetricsResult;

/**
 * @author Sergey Kuptsov
 */
public interface TrainListener extends Serializable {
    void onEpochStarted(int epochId);

    void onEpochFinished(int epochId, Optional<MetricsResult> metrics, Optional<MetricsResult> trainMetrics, MetricsResult costMetrics);

    void onBatchStarted(int batchId);

    void onBatchFinished(int batchId);

    void onTrainStarted();

    void onTrainFinished();
}
