package ru.kuptservol.jml.train.listener;


import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public class LogTrainListener implements TrainListener {
    private final static Logger logger = LoggerFactory.getLogger(LogTrainListener.class);

    @Override
    public void onEpochStarted(int epochId) {
    }

    @Override
    public void onEpochFinished(int epochId,
            Optional<ResultHandler> trainDataMetrics,
            Optional<ResultHandler> testDataMetrics,
            ResultHandler trainDataCost,
            Optional<ResultHandler> testDataCost)
    {
        logger.info("Epoch {} {} {} {} {}",
                epochId,
                trainDataMetrics.map(ResultHandler::print).orElse(""),
                testDataMetrics.map(ResultHandler::print).orElse(""),
                trainDataCost.print(),
                testDataCost.map(ResultHandler::print).orElse(""));
    }

    @Override
    public void onBatchStarted(int batchId) {
    }

    @Override
    public void onBatchFinished(int batchId) {
    }

    @Override
    public void onTrainStarted() {
        logger.info("Train started");
    }

    @Override
    public void onTrainFinished(Model m) {
        logger.info("Train finished");
    }
}
