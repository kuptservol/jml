package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.SimpleModel;

/**
 * @author Sergey Kuptsov
 */
public class SimpleModelTest {

    @Test
    public void test() {
        Tensor W1 = Tensor.tensor(new double[][]{
                {0.1, 0.2, 0.3},
                {0.4, 0.5, 0.6},
                {0.7, 0.8, 0.9},
                {0.10, 0.11, 0.12}});

        Tensor b1 = Tensor.tensor(new double[]{0.13, 0.14, 0.15});

        Tensor W2 = Tensor.tensor(new double[][]{
                {0.12},
                {0.43},
                {0.72}});

        Tensor b2 = Tensor.tensor(new double[]{0.77});

        Tensor input = Tensor.tensor(new double[]{0.34, 0.23, 0.5, 0.9});

        Tensor Y = Tensor.tensor(5);

        SimpleModel model = new SimpleModel(W1, b1, W2, b2);

        Tensor forward = model.forward(input, Y);

        System.out.println("LOSS: \n " + forward);
        Assertions.assertNear(Tensor.tensor(14.0285), forward, 0.0001);

        model.backward();

        System.out.println("W1_grad: \n" + W1.grad);
        Assertions.assertNear(W1.grad, Tensor.tensor(new double[][]{
                {-0.3056, -1.0952, -1.8338},
                {-0.2067, -0.7409, -1.2405},
                {-0.4495, -1.6105, -2.6967},
                {-0.8090, -2.8990, -4.8541}}), 0.0001);

        System.out.println("B1_grad: \n" + b1.grad);
        Assertions.assertNear(Tensor.tensor(new double[]{-0.8989, -3.2211, -5.3935}), b1.grad, 0.0001);

        System.out.println("W2_grad: \n" + W2.grad);
        Assertions.assertNear(Tensor.tensor(new double[][]{
                {-1.4682},
                {-2.4121},
                {-3.3559}}), W2.grad, 0.0001);

        System.out.println("B2_grad: \n" + b2.grad);
        Assertions.assertNear(Tensor.tensor(-7.4909), b2.grad, 0.0001);
    }
}
