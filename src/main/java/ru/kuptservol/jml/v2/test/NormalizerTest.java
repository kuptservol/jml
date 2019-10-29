package ru.kuptservol.jml.v2.test;

import java.io.IOException;
import java.nio.file.Paths;

import org.junit.Test;
import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.data.DataSets;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.Init;

/**
 * @author Sergey Kuptsov
 */
public class NormalizerTest {

    @Test
    public void test() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        Tensor x_train = Tensor.tensor(mnist.train.x);
        Tensor x_valid = Tensor.tensor(mnist.validation.get().x);

        System.out.println(x_train.mean());
        System.out.println(x_train.std());

        Tensor mean = x_train.mean();
        double std = x_train.std();

        x_train = Init.MeanNormalizer.normalize(x_train, mean, std);
        System.out.println(x_train.mean());
        System.out.println(x_train.std());

        Assertions.assertNearZero(x_train.mean().sumD());
        Assertions.assertNearZero(1 - x_train.std());

        x_valid = Init.MeanNormalizer.normalize(x_valid, mean, std);
        System.out.println(x_valid.mean());
        System.out.println(x_valid.std());

        Assertions.assertNearZero(x_valid.mean().sumD());
        Assertions.assertNearZero(1 - x_valid.std());
    }
}
