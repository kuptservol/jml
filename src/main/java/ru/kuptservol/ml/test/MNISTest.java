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
import ru.kuptservol.ml.metric.result.Metrics;
import ru.kuptservol.ml.metric.result.PlotGraphMetric;
import ru.kuptservol.ml.model.Model;
import ru.kuptservol.ml.model.Models;
import ru.kuptservol.ml.result.function.OutputFunctions;
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
                .resultF(OutputFunctions.MAX_INDEX)
                .metrics(ru.kuptservol.ml.metric.Metrics.ACCURACY.build())
                .build();

        logger.debug("Accuracy before learn: " + ru.kuptservol.ml.metric.Metrics.ACCURACY.build().execute(model, mnist.train.x, mnist.train.y).print());

        model.train(mnist);

        Path path = Paths.get("/tmp/mnist_model_with_defaults");

        model.save(path);
        Model modelLoaded = Models.load(path);

        logger.debug("X0: " + M.asPixels(M.to(mnist.train.x[1], 28, 28)));
        logger.debug("Expected answer: " + modelLoaded.resultF.process(mnist.train.y[1]));
        logger.debug("Model answer: " + modelLoaded.output(mnist.train.x[1]));
    }

    @Test
    public void learnWithDefaultSettingsWithCostGraph() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/tmp/mnist"));

        PlotGraphMetric graph = new PlotGraphMetric();

        Model model = Models.linear(784, 30, 10)
                .trainer(Trainers.SGD(100, 3).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .costFunction(CostFunctions.MSE.metric(Metrics.GRAPH_AND_LOG(graph)).build())
                .metrics(ru.kuptservol.ml.metric.Metrics.ACCURACY.build())
                .build();

        model.train(mnist);

        graph.save(Paths.get("/tmp/mnist/cost_graph_res"));
    }

    @Test
    public void learnWithCrossEntropy() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/tmp/mnist"));

        PlotGraphMetric graph = new PlotGraphMetric();

        Model model = Models.linear(784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .costFunction(CostFunctions.CROSS_ENTROPY.metrics(Metrics.GRAPH_AND_LOG(graph)).build())
                .metrics(ru.kuptservol.ml.metric.Metrics.ACCURACY.build())
                .build();

        model.train(mnist);

        graph.save(Paths.get("/tmp/mnist/cost_graph_res"));
    }
}
