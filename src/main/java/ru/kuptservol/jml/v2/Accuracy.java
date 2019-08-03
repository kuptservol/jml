package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class Accuracy implements Metrics {

    @Override
    public double calc(Tensor pred, Tensor y) {
        double correct = 0;


        return correct / pred.shape[0];
    }
}
