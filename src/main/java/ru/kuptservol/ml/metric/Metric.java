package ru.kuptservol.ml.metric;

import java.io.Serializable;

import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface Metric extends Serializable {

    ru.kuptservol.ml.metric.result.Metric execute(Model m, double[][] X, double[][] Y);
}
