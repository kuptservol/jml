package ru.kuptservol.ml.result.function;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface OutputFunction extends Serializable {

    double process(double[] activations);
}
