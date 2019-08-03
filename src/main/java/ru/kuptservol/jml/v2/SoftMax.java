package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 *
 * numerically stable handling overflow for sum of exp calc with big values
 */
public class SoftMax implements Layer {
    private Tensor inp;
    private Tensor out;
    private final int dim;

    public SoftMax(int dim) {
        this.dim = dim;
    }

    public SoftMax() {
        this.dim = 0;
    }

    @Override
    public Tensor forward(Tensor inp) {
        this.inp = inp;
        Tensor max = inp.max(dim);
        Tensor expStable = inp.minus(max.broadcast(inp)).exp();

        out = expStable.div(expStable.sum(1).broadcast(inp));
        return out;
    }

    @Override
    /*
    //todo
    i=j: pi*(1-pj)
    i!=j:-pi*pj
     */
    public void backward() {
        inp.grad = out.grad.mul(inp.F(v -> v < 0 ? 0 : 1.0));
    }
}
