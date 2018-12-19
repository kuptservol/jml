package ru.kuptservol.ml.activation.function;

import ru.kuptservol.ml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class Sigmoid implements ActivationFunction {

    @Override
    public double[] activate(double[] z) {
        return M.FR(this::sigmoid, z);
    }

    @Override
    public double[] dADz(double[] z) {
        return M.FR(v -> sigmoid(v) * (1 - sigmoid(v)), z);
    }

    private double sigmoid(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }
}
