package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.LogSoftMax;

/**
 * @author Sergey Kuptsov
 */
public class LogSoftMaxTest {

    @Test
    public void test() {
        Tensor pred = Tensor.tensor(new double[]{-0.1364, 0.0806, -0.1736, -0.0598, -0.1254, -0.0053, -0.0437, 0.0530,
                -0.0881, -0.0052});

        LogSoftMax logSoftMax = new LogSoftMax(0);

        Tensor sm_pred = logSoftMax.forward(pred);

        System.out.println(sm_pred);
        Assertions.assertNear(
                Tensor.tensor(new double[]{-2.3917, -2.1747, -2.4289, -2.3151, -2.3807, -2.2606, -2.2989, -2.2023, -2.3434, -2.2605}),
                sm_pred, 0.001);

        pred = Tensor.tensor(new double[][]
                {
                        {-0.1364, 0.0806, -0.1736, -0.0598, -0.1254, -0.0053, -0.0437, 0.0530, -0.0881, -0.0052},
                        {-0.1107, -0.0043, -0.2408, -0.0322, -0.0979, -0.0360, -0.0579, 0.0704, -0.1720, 0.0671}
                });

        System.out.println();
        logSoftMax = new LogSoftMax(1);
        sm_pred = logSoftMax.forward(pred);
        System.out.println(sm_pred);

        Assertions.assertNear(
                Tensor.tensor(new double[][]{
                        {-2.3917, -2.1747, -2.4289, -2.3151, -2.3807, -2.2606, -2.2990, -2.2023, -2.3434, -2.2605},
                        {-2.3562, -2.2498, -2.4863, -2.2777, -2.3434, -2.2815, -2.3034, -2.1751, -2.4175, -2.1784}
                }),
                sm_pred, 0.001);
    }
}
