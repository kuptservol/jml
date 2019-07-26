package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.Relu0Mean;

/**
 * @author Sergey Kuptsov
 */
public class ReluTest {

    @Test
    public void testGrad() {
        Tensor inp = Tensor.tensor(new double[][]{
                {6.1, 7.2, 8.3, -9.4, -10.5}});

        Relu0Mean relu = new Relu0Mean();

        Tensor reluV = relu.forward(inp);
        System.out.println(reluV);
        Assertions.assertNear(Tensor.tensor(new double[]{5.6000, 6.7000, 7.8000, -0.5000, -0.5000}), reluV, 0.0001);

        reluV.grad = Tensor.tensor(new double[]{0.16, 0.17, 0.18, 0.19, 0.2});
        relu.backward();
        System.out.println(inp.grad);

        Assertions.assertNear(Tensor.tensor(new double[]{0.1600, 0.1700, 0.1800, 0.0000, 0.0000}), inp.grad, 0.0001);
    }

}
