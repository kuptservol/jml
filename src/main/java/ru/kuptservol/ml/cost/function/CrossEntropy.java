package ru.kuptservol.ml.cost.function;

import ru.kuptservol.ml.metric.result.MetricsResult;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class CrossEntropy implements CostFunction {

    @Override
    public MetricsResult execute(Model m, double[][] trainX, double[][] trainY) {
        return null;
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return new double[0];
    }
}
