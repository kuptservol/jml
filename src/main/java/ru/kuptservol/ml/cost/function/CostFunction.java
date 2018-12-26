package ru.kuptservol.ml.cost.function;

import java.io.Serializable;

import ru.kuptservol.ml.metric.result.Metric;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface CostFunction extends Serializable {

    Metric cost(Model m, double[][] trainX, double[][] trainY);

    /**
     * return Dcost/dA
     */
    double[] backprop(double[] activations, double[] y);
}
