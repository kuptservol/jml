package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.NDTensor;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.MSE;

/**
 * @author Sergey Kuptsov
 */
public class MSETest {

    @Test
    public void testGrad() {
        Tensor Y = Tensor.tensor(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});

        Tensor inp = Tensor.tensor(new double[][]{
                {6.1, 7.2, 8.3, 9.4, 10.5}});

        MSE loss = new MSE();

        Tensor lossV = loss.forward(inp, Y);
        System.out.println(lossV);
        Assertions.assertNear(Tensor.tensor(28.1100), lossV, 0.0001);

        loss.backward();
        System.out.println(inp.grad);

        Assertions.assertNear(Tensor.tensor(new double[][]{{2.0400, 2.0800, 2.1200, 2.1600, 2.2000}}), inp.grad, 0.0001);
    }

    @Test
    public void testNDGrad() {
        NDTensor Y = NDTensor.tensor(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});

        NDTensor inp = NDTensor.tensor(new double[]{6.1, 7.2, 8.3, 9.4, 10.5});

//        NDTensor loss = inp.minus(Y).pow(2).mean();
        NDTensor loss = inp.minus(Y);

        System.out.println(loss.val());
//        Assertions.assertNear(Nd4j.createFromArray(28.1100), arr, 0.0001);

        loss.backward();
//        System.out.println(inp.grad);

//        Assertions.assertNear(Tensor.tensor(new double[][]{{2.0400, 2.0800, 2.1200, 2.1600, 2.2000}}), inp.grad, 0.0001);
        System.out.println(loss.grad());

    }

}
