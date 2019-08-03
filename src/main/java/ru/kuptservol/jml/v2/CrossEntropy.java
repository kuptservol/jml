package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 * negaitive log likelihood with softmax probability
 */
public class CrossEntropy implements Loss {

    private Tensor inp;
    private Tensor Y;
    private final int dim;
    private final SoftMax softMax;
    private Tensor predProb;

    public CrossEntropy(int dim) {
        this.dim = dim;
        this.softMax = new SoftMax(dim);
    }

    public CrossEntropy() {
        this(0);
    }

    @Override
    public Tensor forward(Tensor pred, Tensor Y) {
        this.inp = pred;
        this.Y = Y;
        this.predProb = this.softMax.forward(pred);

//        if (dim == 0) {
        return Y.mul(predProb.log()).mean().neg();
//        } else {
//            //for wrapped one-hot encoded vectors
//            throw new NotImplementedException();
//        }
    }

    @Override
    public void backward() {
        inp.grad = predProb.minus(Y);
    }
}
