package ru.kuptservol.jml.weight.initializer;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface WeightInitializer extends Serializable {

    double[][] init(int x, int y);

    double[] init(int x);
}
