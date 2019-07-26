package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class Linear implements Layer{

    private Tensor inp;
    private Tensor out;
    private final Tensor W;
    private final Tensor b;

    public Linear(Tensor w, Tensor b) {
        this.W = w;
        this.b = b;
    }

    @Override
    public Tensor forward(Tensor inp) {
        this.inp = inp;
        out = inp.mmul(W).plus(b);

        return out;
    }

    @Override
    public void backward() {
        inp.grad = out.grad.mmul(W.T());
        W.grad = inp.T().mmul(out.grad);
        b.grad = out.grad;
    }
}
