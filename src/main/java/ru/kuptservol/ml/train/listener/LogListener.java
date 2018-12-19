package ru.kuptservol.ml.train.listener;


import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.kuptservol.ml.metric.MetricsResult;

/**
 * @author Sergey Kuptsov
 */
public class LogListener implements TrainListener {
    private final static Logger logger = LoggerFactory.getLogger(LogListener.class);

    @Override
    public void onEpochStarted(int epochId) {
    }

    @Override
    public void onEpochFinished(int epochId, Optional<MetricsResult> trainMetrics, Optional<MetricsResult> testMetrics, MetricsResult costMetrics) {
        logger.info("Epoch {} {} {} {}",
                epochId,
                trainMetrics.map(tM -> "train " + tM.print()).orElse(""),
                testMetrics.map(tM -> "test " + tM.print()).orElse(""),
                costMetrics.print());
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
    public void onTrainFinished() {
        logger.info("Train finished");
    }
}
