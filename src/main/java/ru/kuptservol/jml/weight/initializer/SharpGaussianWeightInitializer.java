package ru.kuptservol.jml.weight.initializer;

import java.util.Random;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 * makes initials weights mean and std to be close to 0 - so sigmoid not saturates
 */
public class SharpGaussianWeightInitializer implements WeightInitializer {

    private final Random random = new Random();

    @Override
    public double[][] initWeights(int x, int y) {
        double[][] vals = new double[x][y];
        final double divisor = 1 / Math.sqrt(x);

        return M.FR(v -> divisor * random.nextGaussian(), vals);
    }

    @Override
    public double[] initBiases(int x) {
        double[] vals = new double[x];

        return M.FR(v -> random.nextGaussian(), vals);
    }
}
