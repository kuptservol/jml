package ru.kuptservol.jml.weight.initializer;

/**
 * @author Sergey Kuptsov
 */
public class WeightInitializers {

    public static GaussianWeightInitializer GAUSSIAN(double limit) {
        return new GaussianWeightInitializer(limit);
    }

    public static SharpGaussianWeightInitializer SHARP_GAUSSIAN = new SharpGaussianWeightInitializer();
}
