package ru.kuptservol.jml.activation.function;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 * RELU that keeps mean to 0
 */
public class ReLU0Mean implements ActivationFunction {

    @Override
    public double[] activate(double[] z) {
        return M.FR(v -> v <= 0 ? 0 : v - 0.5, z);
    }

    @Override
    public double[] dADz(double[] z) {
        return M.FR(v -> v < 0 ? 0 : 1.0, z);
    }
}
