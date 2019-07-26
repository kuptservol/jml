package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 * makes initials weights mean and std to be close to 0 for RELU activations
 */
public class Kaiming implements WeightInitializer {

    @Override
    public Tensor init(int... shape) {
        return Tensor.rand(shape).mul(Math.sqrt(2. / shape[0]));
    }
}
