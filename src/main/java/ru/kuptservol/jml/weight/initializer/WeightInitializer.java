package ru.kuptservol.jml.weight.initializer;

import java.io.Serializable;

/**
 * @author Sergey Kuptsov
 */
public interface WeightInitializer extends Serializable {

    double[][] initWeights(int x, int y);

    double[] initBiases(int x);
}
