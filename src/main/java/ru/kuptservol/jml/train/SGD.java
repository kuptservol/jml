package ru.kuptservol.jml.train;

import java.util.Iterator;
import java.util.Optional;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.jml.cost.function.CostFunction;
import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.layer.Layer;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.model.Model;
import ru.kuptservol.jml.optimization.EarlyStopping;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
/**
 * Stochastic Gradient Descent
 */
public class SGD implements Trainer {

    @Builder.Default
    private int batchSize = 10;
    @Builder.Default
    private long epochs = 30;
    @Builder.Default
    private long parallelization = 1;
    @Builder.Default
    private boolean normalize = false;

    @Override
    public void train(Model m, DataSet dataSet) {
        if (normalize) {
            normalize(dataSet);
        }

        double[][] trainX = dataSet.train.x;
        double[][] trainY = dataSet.train.y;

        m.trainListener.onTrainStarted();

        Iterator<Double> learningRates = m.adaptiveLearningRate.learningRates();
        while (learningRates.hasNext()) {
            Double learningRate = learningRates.next();
            m.trainListener.setLearningRate(learningRate);
            m.layers.setLearningRate(learningRate);
            m.earlyStopO.ifPresent(EarlyStopping::reset);

            for (int i = 0; i < epochs && m.earlyStopO.map(EarlyStopping::doContinue).orElse(true); i++) {
                m.trainListener.onEpochStarted(i);

                M.shuffle(trainX, trainY);

                M.Data[] miniBatches = M.chunk(trainX, trainY, batchSize);

                for (int j = 0; j < miniBatches.length; j++) {
                    M.Data miniBatch = miniBatches[j];

                    trainOnMiniBatch(miniBatch, m, j);
                }

                handleEpochResults(m, dataSet, i);

            }
        }
        m.trainListener.onTrainFinished(m);
    }

    private void normalize(DataSet dataSet) {
        double trainXMean = M.mean(dataSet.train.x);
        double trainXStd = M.std(dataSet.train.x);

        dataSet.train.x = M.normalizeR(dataSet.train.x, trainXMean, trainXStd);

        dataSet.validation.ifPresent(data -> dataSet.validation.get().x = M.normalizeR(data.x, trainXMean, trainXStd));
        dataSet.test.ifPresent(data -> dataSet.test.get().x = M.normalizeR(data.x, trainXMean, trainXStd));
    }

    private void trainOnMiniBatch(M.Data trainMiniBatch, Model m, int batchId) {
        m.trainListener.onBatchStarted(batchId);

        CostFunction costFunction = m.costFunction;

        m.layers.forEach(Layer::onBatchStarted);

        for (int i = 0; i < trainMiniBatch.size; i++) {
            double[] activations = m.layers.forward(trainMiniBatch.x[i]);

            double[] dCostDa = costFunction.backprop(m.layers.last(), activations, trainMiniBatch.y[i]);

            m.layers.backprop(dCostDa);
        }

        m.layers.forEach(layer -> layer.onBatchFinished(trainMiniBatch.size, m.regularization));

        m.trainListener.onBatchFinished(batchId);
    }

    private void handleEpochResults(Model m, DataSet dataSet, int epoch) {
        // result handlers
        Optional<ResultHandler> trainMetrics = Optional.ofNullable(m.metrics).map(metrics ->
                m.metricResultHandler.wrap(
                        metrics.execute(m, dataSet.train.x, dataSet.train.y),
                        "train",
                        m.metrics.printFormat()));

        Optional<ResultHandler> testMetrics = Optional.empty();
        Optional<ResultHandler> testDataCost = Optional.empty();
        if (dataSet.test.isPresent()) {
            if (Optional.ofNullable(m.metrics).isPresent()) {
                testMetrics = Optional.ofNullable(m.metrics).map(metrics ->
                        m.metricResultHandler.wrap(
                                metrics.execute(m, dataSet.test.get().x, dataSet.test.get().y),
                                "test",
                                m.metrics.printFormat()));
            }

            testDataCost = Optional.of(m.costResultHandler.wrap(
                    m.costFunction.cost(m, dataSet.test.get().x, dataSet.test.get().y),
                    "test",
                    m.costFunction.printFormat()));
        }

        Optional<ResultHandler> validationMetrics = Optional.empty();
        Optional<ResultHandler> validationDataCost = Optional.empty();
        if (dataSet.validation.isPresent()) {
            if (Optional.ofNullable(m.metrics).isPresent()) {
                validationMetrics = Optional.ofNullable(m.metrics).map(metrics ->
                        m.metricResultHandler.wrap(
                                metrics.execute(m, dataSet.validation.get().x, dataSet.validation.get().y),
                                "validation",
                                m.metrics.printFormat()));
            }

            validationDataCost = Optional.of(m.costResultHandler.wrap(
                    m.costFunction.cost(m, dataSet.validation.get().x, dataSet.validation.get().y),
                    "validation",
                    m.costFunction.printFormat()));
        }

        ResultHandler trainDataCost = m.costResultHandler.wrap(
                m.costFunction.cost(m, dataSet.train.x, dataSet.train.y),
                "train",
                m.costFunction.printFormat());

        m.earlyStopO.ifPresent(earlyStop -> {
                    if (Optional.ofNullable(m.metrics).isPresent()) {
                        if (dataSet.validation.isPresent())
                            earlyStop.countSerie(m.metrics.execute(m, dataSet.validation.get().x, dataSet.validation.get().y));
                        else dataSet.test.ifPresent(data -> earlyStop.countSerie(m.metrics.execute(m, data.x, data.y)));
                    } else {
                        if (dataSet.validation.isPresent())
                            earlyStop.countSerie(1 / m.costFunction.cost(m, dataSet.validation.get().x, dataSet.validation.get().y));
                        else dataSet.test.ifPresent(data -> earlyStop.countSerie(1 / m.costFunction.cost(m, data.x, data.y)));
                    }
                }
        );

        m.trainListener.onEpochFinished(epoch, trainMetrics, testMetrics, validationMetrics, trainDataCost, testDataCost, validationDataCost);
    }
}
