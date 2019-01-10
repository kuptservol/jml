package ru.kuptservol.jml.activation.function;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface ActivationFunction extends Serializable {

    double[] activate(double[] values);

    double[] dADz(double[] z);
}
