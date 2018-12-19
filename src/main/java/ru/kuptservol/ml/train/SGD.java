package ru.kuptservol.ml.train;

import java.util.Optional;

import ru.kuptservol.ml.cost.function.CostFunction;
import ru.kuptservol.ml.data.DataSet;
import ru.kuptservol.ml.layer.Layer;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.metric.MetricsResult;
import ru.kuptservol.ml.model.Model;
import lombok.AllArgsConstructor;
import lombok.Builder;

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

            Optional<MetricsResult> trainMetrics = Optional.ofNullable(m.metrics).map(metrics -> metrics.execute(m, trainX, trainY));
            Optional<MetricsResult> testMetrics = Optional.empty();
            if (dataSet.test.isPresent()) {
                if (Optional.ofNullable(m.metrics).isPresent()) {
                    testMetrics = Optional.ofNullable(m.metrics).map(metrics -> metrics.execute(m, dataSet.test.get().x, dataSet.test.get().y));
                }
            }

            MetricsResult costMetrics = m.costFunction.execute(m, trainX, trainY);

            m.trainListener.onEpochFinished(i, trainMetrics, testMetrics, costMetrics);
        }
        m.trainListener.onTrainFinished();
    }

    private void trainOnMiniBatch(M.Data trainMiniBatch, Model m, int batchId) {
        m.trainListener.onBatchStarted(batchId);

        CostFunction costFunction = m.costFunction;

        m.layers.forEach(Layer::onBatchStarted);

        for (int i = 0; i < trainMiniBatch.size; i++) {
            double[] activations = m.layers.forward(trainMiniBatch.x[i]);
            m.layers.backprop(costFunction.backprop(activations, trainMiniBatch.y[i]));
        }

        m.layers.forEach(layer -> layer.onBatchFinished(trainMiniBatch.size));

        m.trainListener.onBatchFinished(batchId);
    }
}
