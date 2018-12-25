package ru.kuptservol.ml.metric;

import java.io.Serializable;

import ru.kuptservol.ml.metric.result.MetricsResult;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface Metric extends Serializable {

    MetricsResult execute(Model m, double[][] X, double[][] Y);
}
