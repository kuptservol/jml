package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class MeanNormalizer implements Normalizer {

    @Override
    public Tensor normalize(Tensor x, Tensor mean, double std) {
        return x.minus(mean).div(std);
    }
}
