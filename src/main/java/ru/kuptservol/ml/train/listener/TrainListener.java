package ru.kuptservol.ml.train.listener;

import java.io.Serializable;
import java.util.Optional;
import java.util.function.Consumer;

import ru.kuptservol.ml.metric.result.MetricsResult;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface TrainListener extends Serializable {
    void onEpochStarted(int epochId);

    void onEpochFinished(int epochId, Optional<MetricsResult> trainMetrics, Optional<MetricsResult> testMetrics, MetricsResult costMetrics);

    void onBatchStarted(int batchId);

    void onBatchFinished(int batchId);

    void onTrainStarted();

    void onTrainFinished(Model m);
}
