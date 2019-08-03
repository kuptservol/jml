package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 *
 * optimized with LogSumExp trick - numerically stable handling overflow for sum of exp calc with big values
 */
public class LogSoftMax implements Layer {
    private Tensor inp;
    private Tensor out;
    private final int dim;

    public LogSoftMax(int dim) {
        this.dim = dim;
    }

    public LogSoftMax() {
        this.dim = 0;
    }

    @Override
    public Tensor forward(Tensor inp) {
        this.inp = inp;
        out = inp.minus(logsumexp(inp).broadcast(inp));
        return out;
    }

    public Tensor logsumexp(Tensor t) {
        Tensor max = t.max(dim);
        return max.plus(t.minus(max.broadcast(t)).exp().sum(1).log());
    }

    @Override
    /*
    i=j: (1/pi)* pi*(1-pj)
    i!=j:(1/pi)* -pi*pj
     */
    public void backward() {
        inp.grad = out.grad.mul(inp.F(v -> v < 0 ? 0 : 1.0));
    }
}
