package ru.kuptservol.ml.weight.initializer;

import java.util.Random;

import ru.kuptservol.ml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class GaussianWeightInitializer implements WeightInitializer {

    private final double limit;
    private final Random random = new Random();

    GaussianWeightInitializer(double limit) {
        this.limit = limit;
    }

    @Override
    public double[][] init(int x, int y) {
        double[][] vals = new double[x][y];

        return M.FR(v -> limit * random.nextGaussian(), vals);
    }

    @Override
    public double[] init(int x) {
        double[] vals = new double[x];

        return M.FR(v -> limit * random.nextGaussian(), vals);
    }
}
