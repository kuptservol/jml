package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public interface Metrics {
    double calc(Tensor pred, Tensor y);
}
