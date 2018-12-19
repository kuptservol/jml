package ru.kuptservol.ml.weight.initializer;

/**
 * @author Sergey Kuptsov
 */
public class WeightInitializers {

    public static GaussianWeightInitializer gaussian(double limit) {
        return new GaussianWeightInitializer(limit);
    }
}
