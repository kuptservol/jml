package ru.kuptservol.jml.v2;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class SimpleModel {

    private Layer[] layers;
    private MSE loss;

    public SimpleModel(Tensor W1, Tensor b1, Tensor W2, Tensor b2) {
        this.layers = new Layer[]{new Linear(W1, b1), new Relu0Mean(), new Linear(W2, b2)};
        this.loss = new MSE();
    }

    public Tensor forward(Tensor x, Tensor Y) {
        for (Layer layer : layers) {
            x = layer.forward(x);
        }

        return loss.forward(x, Y);
    }

    public void backward() {
        loss.backward();
        for (int i = layers.length - 1; i >= 0; i--) {
            layers[i].backward();
        }
    }
}
