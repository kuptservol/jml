package ru.kuptservol.ml.metric.result;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
@FunctionalInterface
public interface MetricsResult extends Serializable {

    default String print() {
        return "";
    }

    MetricsResult create(double cost, String pattern);
}
