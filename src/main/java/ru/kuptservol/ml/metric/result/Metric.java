package ru.kuptservol.ml.metric.result;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
@FunctionalInterface
public interface Metric extends Serializable {

    default String print() {
        return "";
    }

    Metric create(double cost, String pattern);
}
