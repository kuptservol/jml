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
        Tensor a = Tensor.randn(512, 512);
        Tensor x = Tensor.randn(512);

        for (int i = 0; i < 100; i++) {
            x = relu(x.mmul(a));
        }

        // std of 100 matrix mult of radn init goes to Nan
        System.out.println("Mean: " + x.meanD());
        System.out.println("Std: " + x.std());

        a = Tensor.randn(512, 512).mul(0.01);
        x = Tensor.randn(512);

        for (int i = 0; i < 100; i++) {
            x = relu(x.mmul(a));
        }

        // std of 100 matrix mult of radn*0.01 vanishes
        System.out.println("Mean: " + x.meanD());
        System.out.println("Std: " + x.std());

        a = Init.Kaiming.init(512, 512);
        x = Tensor.randn(512);

        Assertions.assertNear(a.meanD(), 0, 0.01);

        for (int i = 0; i < 100; i++) {
            x = relu(x.mmul(a));
        }

        // std of 100 matrix mult with kaiming init is stable
        System.out.println("Mean: " + x.meanD());
        System.out.println("Std: " + x.std());

        Assertions.assertNear(x.meanD(), 0, 0.5);
        Assertions.assertNear(x.std(), 1, 0.5);
    }

    private Tensor relu(Tensor t) {
        return t.clamp_min(0.0).minus(0.5);
    }
}
