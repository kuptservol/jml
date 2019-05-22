package ru.kuptservol.jml.weight.initializer;

/**
 * @author Sergey Kuptsov
 */
public class WeightInitializers {

    public static GaussianWeightInitializer Gaussian(double limit) {
        return new GaussianWeightInitializer(limit);
    }

    public static SharpGaussianWeightInitializer SharpGaussian = new SharpGaussianWeightInitializer();
}
