package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 * Initialize weights so mean=0 and std=1 keeps after multiple relu(a @ x) operations
 */
public class Kaiming implements WeightInitializer {

    @Override
    public Tensor init(int... shape) {
        return Tensor.randn(shape).mul(Math.sqrt(2. / shape[0]));
    }
}
