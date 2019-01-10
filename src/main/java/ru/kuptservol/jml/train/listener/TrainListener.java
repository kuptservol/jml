package ru.kuptservol.jml.train.listener;

import java.io.Serializable;
import java.util.Optional;

import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface TrainListener extends Serializable {
    void onEpochStarted(int epochId);

    void onEpochFinished(int epochId,
            Optional<ResultHandler> trainDataMetrics,
            Optional<ResultHandler> testDataMetrics,
            ResultHandler trainDataCost,
            Optional<ResultHandler> testDataCost);

    void onBatchStarted(int batchId);

    void onBatchFinished(int batchId);

    void onTrainStarted();

    void onTrainFinished(Model m);
}
