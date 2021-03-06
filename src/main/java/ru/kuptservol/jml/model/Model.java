package ru.kuptservol.jml.model;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.Optional;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.cost.function.CostFunction;
import ru.kuptservol.jml.cost.function.CostFunctions;
import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.layer.Layers;
import ru.kuptservol.jml.metric.Metric;
import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.metric.result.ResultHandlers;
import ru.kuptservol.jml.optimization.AdaptiveLearningRate;
import ru.kuptservol.jml.optimization.EarlyStopping;
import ru.kuptservol.jml.optimization.Optimizations;
import ru.kuptservol.jml.optimization.Optimizers;
import ru.kuptservol.jml.optimization.Regularization;
import ru.kuptservol.jml.result.function.OutputFunction;
import ru.kuptservol.jml.result.function.OutputFunctions;
import ru.kuptservol.jml.train.SGD;
import ru.kuptservol.jml.train.Trainer;
import ru.kuptservol.jml.train.listener.TrainListener;
import ru.kuptservol.jml.train.listener.TrainListeners;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
@Getter
public class Model implements Serializable {

    @Builder.Default
    public Layers layers = Layers
            .fullyConnected(
                    0,
                    WeightInitializers.Gaussian(1),
                    ActivationFunctions.Sigmoid,
                    Optimizers.None(),
                    10,
                    1)
            .build();

    @Builder.Default
    private Trainer trainer = SGD
            .builder()
            .build();

    @Builder.Default
    public Regularization regularization = Optimizations.L2Reg(0.0);
    @Builder.Default
    public TrainListener trainListener = TrainListeners.LOG_LISTENER;
    @Builder.Default
    public CostFunction costFunction = CostFunctions.MSE.build();
    @Builder.Default
    public Metric metrics;
    @Builder.Default
    public OutputFunction resultF = OutputFunctions.MaxVal;
    @Builder.Default
    public ResultHandler costResultHandler = ResultHandlers.LOG;
    @Builder.Default
    public ResultHandler metricResultHandler = ResultHandlers.Empty;
    @Builder.Default
    public Optional<EarlyStopping> earlyStopO = Optional.empty();
    @Builder.Default
    public AdaptiveLearningRate adaptiveLearningRate = Optimizations.ConstLearningRate(0.1);

    public void train(double[][] X, double[][] Y) {
        trainer.train(this, X, Y);
    }

    public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY) {
        trainer.train(this, trainX, trainY, testX, testY);
    }

    public double[] forward(double[] x) {
        return layers.forward(x);
    }

    public double output(double[] x) {
        return resultF.process(forward(x));
    }

    public void train(DataSet dataSet) {
        trainer.train(this, dataSet);
    }

    public void save(Path to) throws IOException {
        Models.save(to, this);
    }
}


