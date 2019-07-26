package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public interface Layer {

    Tensor forward(Tensor inp);

    void backward();
}
