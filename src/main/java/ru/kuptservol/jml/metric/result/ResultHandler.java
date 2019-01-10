package ru.kuptservol.jml.metric.result;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
@FunctionalInterface
public interface ResultHandler extends Serializable {

    default String print() {
        return "";
    }

    ResultHandler wrap(double cost, String dataLabel, String format);
}
