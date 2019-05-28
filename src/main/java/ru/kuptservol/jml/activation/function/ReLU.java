package ru.kuptservol.jml.activation.function;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class ReLU implements ActivationFunction {

    @Override
    public double[] activate(double[] z) {
        return M.FR(v -> v <= 0 ? 0 : v, z);
    }

    @Override
    public double[] dADz(double[] z) {
        return M.FR(v -> v < 0 ? 0 : 1.0, z);
    }
}
