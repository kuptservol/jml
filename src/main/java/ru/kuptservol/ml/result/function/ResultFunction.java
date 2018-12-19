package ru.kuptservol.ml.result.function;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface ResultFunction extends Serializable {

    double apply(double[] activations);
}
