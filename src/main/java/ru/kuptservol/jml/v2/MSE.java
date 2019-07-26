package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class MSE {

    private Tensor inp;
    private Tensor Y;

    public Tensor forward(Tensor inp, Tensor Y) {
        this.inp = inp;
        this.Y = Y;
        return inp.minus(Y).pow(2).mean();
    }

    public void backward() {
        inp.grad = inp.minus(Y).mul(2.0).div(Y.shape[0]);
    }
}
