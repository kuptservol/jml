package ru.kuptservol.jml.v2;

import java.io.Serializable;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public interface WeightInitializer extends Serializable {

    Tensor init(int... shape);
}
