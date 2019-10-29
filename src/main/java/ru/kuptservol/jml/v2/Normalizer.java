package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public interface Normalizer {

    Tensor normalize(Tensor x, Tensor mean, double std);
}
