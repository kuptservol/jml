package ru.kuptservol.jml.tensor;

import java.util.Arrays;

import org.junit.Test;

/**
 * @author Sergey Kuptsov
 */
public class TensorTest {

    @Test
    public void createScalar() {
        Tensor tensor = Tensor.tensor(1);

        System.out.println(tensor);
        System.out.println(Arrays.toString(tensor.shape()));
    }

    @Test
    public void createVector() {
        Tensor tensor = Tensor.tensor(new double[]{1, 2, 3});

        System.out.println(tensor);
        System.out.println(Arrays.toString(tensor.shape()));
    }

    @Test
    public void createMatrix() {
        Tensor tensor = Tensor.tensor(new double[][]{{1, 2, 3}, {1, 2, 3}});

        System.out.println(tensor);
        System.out.println(Arrays.toString(tensor.shape()));
    }
}
