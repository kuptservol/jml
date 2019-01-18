package ru.kuptservol.jml.optimization;

import java.util.Iterator;

/**
 * @author Sergey Kuptsov
 */
public class ConstDecreasingAdaptiveLearningRate implements AdaptiveLearningRate {

    private double currentLearningRate;
    private final double stopOnLearningRate;
    private final int divisionStep;

    public ConstDecreasingAdaptiveLearningRate(
            double initialLearningRate,
            double stopOnLearningRate,
            int divisionStep)
    {
        this.currentLearningRate = initialLearningRate;
        this.divisionStep = divisionStep;
        this.stopOnLearningRate = stopOnLearningRate;
    }

    public ConstDecreasingAdaptiveLearningRate(double initialLearningRate) {
        this.currentLearningRate = initialLearningRate;
        this.divisionStep = 2;
        this.stopOnLearningRate = initialLearningRate / 128;
    }

    @Override
    public Iterator<Double> learningRates() {
        return new Iterator<Double>() {

            @Override
            public boolean hasNext() {
                return currentLearningRate > stopOnLearningRate;
            }

            @Override
            public Double next() {
                double lR = currentLearningRate;
                currentLearningRate /= divisionStep;

                return lR;
            }
        };
    }
}
