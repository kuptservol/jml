package ru.kuptservol.jml.test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.cost.function.CostFunctions;
import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.data.DataSets;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.metric.Metrics;
import ru.kuptservol.jml.metric.result.PlotGraphResultHandler;
import ru.kuptservol.jml.metric.result.ResultHandlers;
import ru.kuptservol.jml.model.Model;
import ru.kuptservol.jml.model.Models;
import ru.kuptservol.jml.optimization.Optimizations;
import ru.kuptservol.jml.result.function.OutputFunctions;
import ru.kuptservol.jml.train.Trainers;
import ru.kuptservol.jml.train.listener.LogTrainListener;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
public class MNISTest {
    private final static Logger logger = LoggerFactory.getLogger(LogTrainListener.class);

    @Test
    public void learnWithDefaultSettings() throws IOException, ClassNotFoundException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/default_settings.png"));

        Model model = Models.linear(784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .build();

        logger.debug("Accuracy before learn: " + ResultHandlers.LOG.wrap(
                Metrics.ACCURACY.build().execute(model, mnist.train.x, mnist.train.y), "init",
                Metrics.ACCURACY.build().printFormat()).print());

        model.train(mnist);

        Path path = Paths.get("/opt/jml/model/mnist_model_with_defaults");

        model.save(path);
        Model modelLoaded = Models.load(path);

        logger.debug("X0: " + M.asPixels(M.to(mnist.train.x[1], 28, 28)));
        logger.debug("Expected answer: " + modelLoaded.resultF.process(mnist.train.y[1]));
        logger.debug("Model answer: " + modelLoaded.output(mnist.train.x[1]));
    }

    @Test
    public void learnWithDefaultSettingsWithCostGraph() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/cost_graph_res.png"));

        Model model = Models.linear(784, 30, 10)
                .trainer(Trainers.SGD(100, 3).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .costFunction(CostFunctions.MSE.build())
                .costResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .metrics(Metrics.ACCURACY.build())
                .build();

        model.train(mnist);
    }

    @Test
    /**
     * 2018-12-27 02:11:14:530 +0300 [main] INFO Epoch 0 train Accuracy: 51,493 % test Accuracy: 52,710 % Cross entropy: 466068590519857900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,000
     * ...
     * 2018-12-27 02:36:53:257 +0300 [main] INFO Epoch 84 train Accuracy: 89,676 % test Accuracy: 90,550 % Cross entropy: 99871840825684200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,000
     */
    public void learnWithCrossEntropy() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy.png"));

        Model model = Models.linear(0.05, 784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropyL2Reg() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_l2_reg.png"));

        Model model = Models.linear(0.01, 784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStop() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop.png"));

        Model model = Models.linear(0.01, 784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .earlyStopO(Optional.of(Optimizations.EARLY_STOPPING(5)))
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInit() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init.png"));

        Model model = Models.linear(0.01, WeightInitializers.SHARP_GAUSSIAN, 784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .earlyStopO(Optional.of(Optimizations.EARLY_STOPPING(5)))
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitAdaptiveLR() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_adaptive_lr.png"));

        Model model = Models.linear(0.01, WeightInitializers.SHARP_GAUSSIAN, 784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .earlyStopO(Optional.of(Optimizations.EARLY_STOPPING(3)))
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .adaptiveLearningRate(Optimizations.CONST_DECREASING_LEARNING_RATE(1))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5.png"));

        Model model = Models.linear(0.01, WeightInitializers.SHARP_GAUSSIAN, 0.5, 784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .earlyStopO(Optional.of(Optimizations.EARLY_STOPPING(5)))
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5TanhActivation() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_tanh_af.png"));

        Model model = Models.linear(
                0.01,
                ActivationFunctions.TANH,
                WeightInitializers.SHARP_GAUSSIAN,
                0.5,
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .earlyStopO(Optional.of(Optimizations.EARLY_STOPPING(5)))
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5ReLUActivation() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_relu_af.png"));

        Model model = Models.linear(
                0.01,
                ActivationFunctions.ReLU,
                WeightInitializers.SHARP_GAUSSIAN,
                0.5,
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MAX_INDEX)
                .earlyStopO(Optional.of(Optimizations.EARLY_STOPPING(5)))
                .costFunction(CostFunctions.CROSS_ENTROPY.resultHandler(ResultHandlers.EMPTY).build())
                .metrics(Metrics.ACCURACY.build())
                .metricResultHandler(ResultHandlers.GRAPH_AND_LOG(graph))
                .regularization(Optimizations.L2_REG(5))
                .build();

        model.train(mnist);
    }
}
