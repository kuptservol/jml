package ru.kuptservol.ml.test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.kuptservol.ml.cost.function.CostFunctions;
import ru.kuptservol.ml.data.DataSet;
import ru.kuptservol.ml.data.DataSets;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.metric.Metrics;
import ru.kuptservol.ml.metric.result.MetricsResults;
import ru.kuptservol.ml.metric.result.PlotGraphMetricsResult;
import ru.kuptservol.ml.model.Model;
import ru.kuptservol.ml.model.Models;
import ru.kuptservol.ml.result.function.ResultFunctions;
import ru.kuptservol.ml.train.Trainers;
import ru.kuptservol.ml.train.listener.LogListener;

/**
 * @author Sergey Kuptsov
 */
public class MNISTest {
    private final static Logger logger = LoggerFactory.getLogger(LogListener.class);

    @Test
    public void learnWithDefaultSettings() throws IOException, ClassNotFoundException {
        DataSet mnist = DataSets.MNIST(Paths.get("/tmp/mnist"));

        Model model = Models.linear(784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultFunction(ResultFunctions.MAX_INDEX)
                .metrics(Metrics.ACCURACY.build())
                .build();

        logger.debug("Accuracy before learn: " + Metrics.ACCURACY.build().execute(model, mnist.train.x, mnist.train.y).print());

        model.train(mnist);

        Path path = Paths.get("/tmp/mnist_model_with_defaults");

        model.save(path);
        Model modelLoaded = Models.load(path);

        logger.debug("X0: " + M.asPixels(M.to(mnist.train.x[1], 28, 28)));
        logger.debug("Expected answer: " + modelLoaded.resultFunction.apply(mnist.train.y[1]));
        logger.debug("Model answer: " + modelLoaded.evaluate(mnist.train.x[1]));
    }

    @Test
    public void learnWithDefaultSettingsWithCostGraph() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/tmp/mnist"));

        PlotGraphMetricsResult graph = new PlotGraphMetricsResult();

        Model model = Models.linear(784, 30, 10)
                .trainer(Trainers.SGD(100, 3).build())
                .resultFunction(ResultFunctions.MAX_INDEX)
                .costFunction(CostFunctions.MSE.metricsResult(MetricsResults.GRAPH_AND_LOG(graph)).build())
                .metrics(Metrics.ACCURACY.build())
                .build();

        model.train(mnist);

        graph.save(Paths.get("/tmp/mnist/cost_graph_res"));
    }
}
