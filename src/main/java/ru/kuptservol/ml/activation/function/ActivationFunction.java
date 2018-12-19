package ru.kuptservol.ml.activation.function;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface ActivationFunction extends Serializable {

    double[] activate(double[] values);

    double[] dADz(double[] z);
}
