package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.Linear;

/**
 * @author Sergey Kuptsov
 */
public class LinearTest {

    @Test
    public void testGrad() {
        Tensor inp = Tensor.tensor(new double[][]
                {{0.10, 0.11, 0.12, 0.13}});

        Tensor W = Tensor.tensor(new double[][]{
                {0.1, 0.2, 0.3},
                {0.4, 0.5, 0.6},
                {0.7, 0.8, 0.9},
                {0.10, 0.11, 0.12}});

        Tensor b = Tensor.tensor(new double[]{0.13, 0.14, 0.15});

        Linear linear = new Linear(W, b);

        Tensor forward = linear.forward(inp);
        forward.grad = Tensor.tensor(new double[][]{{0.16, 0.17, 0.18}});

        Assertions.assertNear(Tensor.tensor(new double[][]{{0.2810, 0.3253, 0.3696}}), forward, 0.0001);
        System.out.println(forward);

        linear.backward();

        Assertions.assertNear(Tensor.tensor(new double[][]{
                {0.0160, 0.0170, 0.0180},
                {0.0176, 0.0187, 0.0198},
                {0.0192, 0.0204, 0.0216},
                {0.0208, 0.0221, 0.0234}
        }), W.grad, 0.0001);

        Assertions.assertNear(Tensor.tensor(new double[][]{{0.1600, 0.1700, 0.1800}}), b.grad, 0.0001);

        Assertions.assertNear(Tensor.tensor(new double[][]{{0.1040, 0.2570, 0.4100, 0.0563}}), inp.grad, 0.0001);

        System.out.println("W_grad: \n" + W.grad);
        System.out.println("b_grad: \n" + b.grad);
        System.out.println("inp_grad: \n" + inp.grad);
    }

}
