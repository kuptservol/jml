package ru.kuptservol.jml.activation.function;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class Tanh implements ActivationFunction {

    @Override
    public double[] activate(double[] z) {
        return M.FR(this::tahn, z);
    }

    @Override
    public double[] dADz(double[] z) {
        return M.FR(v -> 1 - Math.pow(tahn(v), 2), z);
    }

    private double tahn(double v) {
        return (Math.exp(v) - Math.exp(-v)) / (Math.exp(v) + Math.exp(-v));
    }
}
