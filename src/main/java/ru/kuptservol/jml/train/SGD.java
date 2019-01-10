package ru.kuptservol.jml.train;

import java.util.Optional;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.jml.cost.function.CostFunction;
import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.layer.Layer;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.model.Model;

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

    @Override
    public void train(Model m, DataSet dataSet) {
        double[][] trainX = dataSet.train.x;
        double[][] trainY = dataSet.train.y;
        m.trainListener.onTrainStarted();
        for (int i = 0; i < epochs; i++) {
            m.trainListener.onEpochStarted(i);

            M.shuffle(trainX, trainY);

            M.Data[] miniBatches = M.chunk(trainX, trainY, batchSize);

            for (int j = 0; j < miniBatches.length; j++) {
                M.Data miniBatch = miniBatches[j];

                trainOnMiniBatch(miniBatch, m, j);
            }

            handleEpochResults(m, dataSet, i);

        }
        m.trainListener.onTrainFinished(m);
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

        ResultHandler trainDataCost = m.costResultHandler.wrap(
                m.costFunction.cost(m, dataSet.train.x, dataSet.train.y),
                "train",
                m.costFunction.printFormat());

        m.trainListener.onEpochFinished(epoch, trainMetrics, testMetrics, trainDataCost, testDataCost);
    }
}
