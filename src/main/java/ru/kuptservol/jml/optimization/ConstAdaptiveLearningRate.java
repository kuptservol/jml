package ru.kuptservol.jml.optimization;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author Sergey Kuptsov
 */
public class ConstAdaptiveLearningRate implements AdaptiveLearningRate {

    private List<Double> lr = new ArrayList<>();

    public ConstAdaptiveLearningRate(double learningRate) {
        lr.add(learningRate);
    }

    @Override
    public Iterator<Double> learningRates() {
        return lr.iterator();
    }
}
