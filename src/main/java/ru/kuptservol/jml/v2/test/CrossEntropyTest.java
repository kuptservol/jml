package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.CrossEntropy;
import ru.kuptservol.jml.v2.LogSoftMax;

/**
 * @author Sergey Kuptsov
 */
public class CrossEntropyTest {

    @Test
    public void test() {

        Tensor pred = Tensor.tensor(new double[][]
                {
                        {-0.1364, 0.0806, -0.1736, -0.0598, -0.1254, -0.0053, -0.0437, 0.0530, -0.0881, -0.0052},
                        {-0.1107, -0.0043, -0.2408, -0.0322, -0.0979, -0.0360, -0.0579, 0.0704, -0.1720, 0.0671}
                });


        LogSoftMax logSoftMax = new LogSoftMax(0);

        System.out.println(logSoftMax.forward(pred));

        Tensor y_one_hot = Tensor.tensor(new double[][]{{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}});

        CrossEntropy crossEntropy = new CrossEntropy(1);

        Tensor loss = crossEntropy.forward(pred, y_one_hot);

        System.out.println(loss);
        Assertions.assertNear(
                Tensor.tensor(0.23084),
                loss,
                0.001);


        crossEntropy.backward();
        System.out.println(pred.grad);
    }
}
