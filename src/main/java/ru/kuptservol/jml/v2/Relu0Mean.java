package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 * RELU that keeps mean to 0
 */
public class Relu0Mean implements Layer {

    private Tensor inp;
    private Tensor out;

    @Override
    public Tensor forward(Tensor inp) {
        this.inp = inp;
        this.out = inp.clamp_min(0.0).minus(0.5);
        return out;
    }

    @Override
    public void backward() {
        inp.grad = out.grad.mul(inp.F(v -> v < 0 ? 0 : 1.0));
    }
}
