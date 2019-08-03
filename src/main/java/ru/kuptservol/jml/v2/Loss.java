package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public interface Loss {
    Tensor forward(Tensor inp, Tensor Y);

    void backward();
}
