package ru.kuptservol.ml.model;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import ru.kuptservol.ml.cost.function.CostFunction;
import ru.kuptservol.ml.cost.function.CostFunctions;
import ru.kuptservol.ml.data.DataSet;
import ru.kuptservol.ml.layer.Layers;
import ru.kuptservol.ml.metric.Metric;
import ru.kuptservol.ml.result.function.ResultFunction;
import ru.kuptservol.ml.result.function.ResultFunctions;
import ru.kuptservol.ml.train.SGD;
import ru.kuptservol.ml.train.Trainer;
import ru.kuptservol.ml.train.listener.TrainListener;
import ru.kuptservol.ml.train.listener.TrainListeners;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
@Getter
public class Model implements Serializable {

    @Builder.Default
    public Layers layers = Layers
            .linear(1, 10, 1)
            .build();

    @Builder.Default
    private Trainer trainer = SGD
            .builder()
            .build();

    @Builder.Default
    public TrainListener trainListener = TrainListeners.LOG_LISTENER;
    @Builder.Default
    public CostFunction costFunction = CostFunctions.MSE.build();
    @Builder.Default
    public Metric metrics;
    @Builder.Default
    public ResultFunction resultFunction = ResultFunctions.MAX_VAL;

    public void train(double[][] X, double[][] Y) {
        trainer.train(this, X, Y);
    }

    public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY) {
        trainer.train(this, trainX, trainY, testX, testY);
    }

    public double evaluate(double[] x) {
        return resultFunction.apply(layers.forward(x));
    }

    public void train(DataSet dataSet) {
        trainer.train(this, dataSet);
    }

    public void save(Path to) throws IOException {
        Models.save(to, this);
    }
}


