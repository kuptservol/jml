package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.Init;

/**
 * @author Sergey Kuptsov
 */
public class KaimingTest {

    @Test
    public void test() {
        Tensor tensor = Init.Kaiming.init(784, 50);

        System.out.println("Mean: " + tensor.meanD());
        System.out.println("std: " + tensor.std());

        Assertions.assertNear(tensor.meanD(), 0, 0.1);
    }
}
