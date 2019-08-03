package ru.kuptservol.jml.v2;

import java.util.stream.IntStream;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class SGD {

    int batchSize = 64;
    int epochs = 1;
    double lr = 0.5;
    Metrics accuracy = new Accuracy();
    CrossEntropy lossF = new CrossEntropy();

    Tensor X;
    Tensor Y;
    Model model;

    public SGD(Tensor x, Tensor y, Model model) {
        X = x;
        Y = y;
        this.model = model;
    }

    public void fit() {
        for (int epoch : IntStream.range(0, epochs).toArray()) {
            Tensor loss = null;
            for (int batch : IntStream.range(0, 1 + X.shape[0] / batchSize).toArray()) {
                int from = batch * batchSize;
                int to = from + batchSize;

                to = to > X.shape[0] - 1 ? X.shape[0] - 1 : to;
                Tensor X_batch = X.getRange(from, to);
                Tensor Y_batch = Y.getRange(from, to);

                loss = lossF.forward(model.forward(X_batch), Y_batch);
                lossF.backward();
                model.backward();

                for (Layer layer : model.getLayers()) {
                    if (layer.getClass().isAssignableFrom(Linear.class)) {
                        Linear linLayer = (Linear) layer;
                        linLayer.W.minusi(linLayer.W.grad.mul(lr));
                        linLayer.b.minusi(linLayer.b.grad.mul(lr));
                    }
                }
            }

            System.out.println("Epoch: " + epoch + "Accuracy : " + accuracy.calc(loss, Y));
        }
    }
}
