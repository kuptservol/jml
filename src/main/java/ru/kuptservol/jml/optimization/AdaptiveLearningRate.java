package ru.kuptservol.jml.optimization;

import java.util.Iterator;

/**
 * @author Sergey Kuptsov
 */
public interface AdaptiveLearningRate {

    Iterator<Double> learningRates();
}
