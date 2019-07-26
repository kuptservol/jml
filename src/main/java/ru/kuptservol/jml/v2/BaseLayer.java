package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public abstract class BaseLayer implements Layer {

    protected Tensor inp;
    protected Tensor out;

    protected BaseLayer(Tensor inp) {
        this.inp = inp;
    }

    @Override
    public Tensor forward(Tensor inp) {
        out = fwd(inp);
        return out;
    }

    public abstract Tensor fwd(Tensor inp);
}
