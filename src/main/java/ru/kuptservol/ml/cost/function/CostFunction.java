package ru.kuptservol.ml.cost.function;

import java.io.Serializable;

import ru.kuptservol.ml.metric.MetricsResult;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface CostFunction extends Serializable {

    MetricsResult execute(Model m, double[][] trainX, double[][] trainY);

    double[] backprop(double[] activations, double[] y);
}
