package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public interface Model {
    Tensor forward(Tensor x);

    void backward();

    Layer[] getLayers();
}
