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
import ru.kuptservol.jml.optimization.Optimizers;
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
                .resultF(OutputFunctions.MaxIndex)
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .build();

        logger.debug("Accuracy before learn: " + ResultHandlers.LOG.wrap(
                Metrics.Accuracy.build().execute(model, mnist.train.x, mnist.train.y), "init",
                Metrics.Accuracy.build().printFormat()).print());

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
                .resultF(OutputFunctions.MaxIndex)
                .costFunction(CostFunctions.MSE.build())
                .costResultHandler(ResultHandlers.GraphAndLog(graph))
                .metrics(Metrics.Accuracy.build())
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
                .resultF(OutputFunctions.MaxIndex)
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
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
                .resultF(OutputFunctions.MaxIndex)
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
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
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInit() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitAdaptiveLR() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_adaptive_lr.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 784, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(3)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .adaptiveLearningRate(Optimizations.ConstDecreasingLearningRate(1))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 0.5, 784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5_baseline() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph" +
                        "/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_baseline" +
                        ".png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 0.5, 784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_rmsprop_0_9() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/test"));

        Model model = Models.linear(
                Models.LinearModelBuilder.builder()
                        .learningRate(0.01)
                        .weightInitializer(WeightInitializers.SharpGaussian)
                        .optimizer(Optimizers.RMSProp(0.9))
                        .build(),
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.MSE.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_adam_0_9_0_9() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_adam_0_9_0_9"));

        Model model = Models.linear(
                Models.LinearModelBuilder.builder()
                        .learningRate(0.01)
                        .weightInitializer(WeightInitializers.SharpGaussian)
                        .optimizer(Optimizers.Adam(0.9, 0.9))
                        .build(),
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.MSE.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void test() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/test"));

        Model model = Models.linear(
                Models.LinearModelBuilder.builder()
                        .learningRate(0.01)
                        .weightInitializer(WeightInitializers.SharpGaussian)
                        .optimizer(Optimizers.Adam(0.9, 0.9))
                        .build(),
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learn_softmax_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_adam_0_9_0_9() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_softmax_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_adam_0_9_0_9"));

        Model model = Models.linear(
                Models.LinearModelBuilder.builder()
                        .learningRate(0.01)
                        .weightInitializer(WeightInitializers.SharpGaussian)
                        .optimizer(Optimizers.Adam(0.9, 0.9))
                        .build(),
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_adam_0_5_0_5() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_rmsprop_0_9"));

        Model model = Models.linear(
                Models.LinearModelBuilder.builder()
                        .learningRate(0.01)
                        .weightInitializer(WeightInitializers.SharpGaussian)
                        .optimizer(Optimizers.Adam(0.5, 0.5))
                        .build(),
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.MSE.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithMSE100L2RegEarlyStopSharpWeightInitWithMomentum_0_5TanhActivation() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_tanh_af.png"));

        Model model = Models.linear(
                0.01,
                ActivationFunctions.Tanh,
                WeightInitializers.SharpGaussian,
                0.5,
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithMSE100L2RegEarlyStopSharpWeightInitWithMomentum_0_5ReLUActivation() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_mse_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_relu_af.png"));

        Model model = Models.linear(
                0.01,
                ActivationFunctions.ReLU,
                WeightInitializers.SharpGaussian,
                0.5,
                784, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.MSE.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5_deep() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_deep.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 0.5, 784, 100, 100, 100, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5_deep_2() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_deep_2.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 0.5, 784, 30, 30, 30, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5_deep_3() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_deep_3.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 0.5, 784, 50, 50, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }

    @Test
    public void learnWithCrossEntropy100L2RegEarlyStopSharpWeightInitWithMomentum_0_5_deep_4() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        PlotGraphResultHandler graph = PlotGraphResultHandler
                .cons(Paths.get("./graph/learn_cross_entropy_100_neurons_l2_reg_early_stop_sharp_weight_init_with_momentum_0_5_deep_4.png"));

        Model model = Models.linear(0.01, WeightInitializers.SharpGaussian, 0.5, 784, 10, 10, 10)
                .trainer(Trainers.SGD(100, 100).build())
                .resultF(OutputFunctions.MaxIndex)
                .earlyStopO(Optional.of(Optimizations.EarlyStopping(5)))
                .costFunction(CostFunctions.CrossEntropy.resultHandler(ResultHandlers.Empty).build())
                .metrics(Metrics.Accuracy.build())
                .metricResultHandler(ResultHandlers.GraphAndLog(graph))
                .regularization(Optimizations.L2Reg(5))
                .build();

        model.train(mnist);
    }
}
