package ru.kuptservol.jml.v2.test;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Sergey Kuptsov
 */
public class NDArrayTest {

    @Test
    public void mse() {
        SameDiff factory = SameDiff.create();
        INDArray Y = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        SDVariable Yvar = factory.var(Y);

        INDArray inp = Nd4j.create(new double[][]{{6.1, 7.2, 8.3, 9.4, 10.5}});
        SDVariable inpVar = factory.var("inpVar", inp);

        SDVariable mse = inpVar.sub(Yvar).pow("aa", 2).mean();

        System.out.println(mse.eval());

        factory.createGradFunction();
        System.out.println(inpVar.gradient().eval());
    }

    @Test
    public void mse2() {
        SameDiff factory1 = SameDiff.create();
        INDArray Y = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        SDVariable Yvar = factory1.var(Y);

        INDArray inp = Nd4j.create(new double[]{6.1, 7.2, 8.3, 9.4, 10.5});
        SameDiff factory2 = Yvar.getSameDiff().min;
        SDVariable inpVar = factory2.var(inp);

        SDVariable mse = inpVar.plus(Yvar);

        System.out.println(mse.eval());
    }
}
