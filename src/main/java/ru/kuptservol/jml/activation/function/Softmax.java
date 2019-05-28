package ru.kuptservol.jml.activation.function;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class Softmax implements ActivationFunction {

    @Override
    public double[] activate(double[] values) {
        double expSums = 0;

        for (double value : values) {
            expSums += Math.exp(value);
        }

        final double expSumsf = expSums;

        return M.FR(value -> Math.exp(value) / expSumsf, values);
    }

    @Override
    public double[] dADz(double[] z) {
        return new double[0];
    }
}
