package ru.kuptservol.jml.activation.function;

import java.util.Arrays;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class StableSoftmax implements ActivationFunction {

    @Override
    public double[] activate(double[] values) {
        double expSums = 0;
        double shift = Arrays.stream(values).max().getAsDouble();

        for (double value : values) {
            expSums += Math.exp(value - shift);
        }

        final double expSumsf = expSums;

        return M.FR(value -> Math.exp(value - shift) / expSumsf, values);
    }

    @Override
    public double[] dADz(double[] z) {
        return new double[0];
    }
}
