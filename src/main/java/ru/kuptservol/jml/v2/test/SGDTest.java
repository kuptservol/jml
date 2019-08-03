package ru.kuptservol.jml.v2.test;

import java.io.IOException;
import java.nio.file.Paths;

import org.junit.Test;
import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.data.DataSets;
import ru.kuptservol.jml.tensor.Tensor;
import ru.kuptservol.jml.v2.Init;
import ru.kuptservol.jml.v2.Layer;
import ru.kuptservol.jml.v2.Linear;
import ru.kuptservol.jml.v2.Model;
import ru.kuptservol.jml.v2.Relu0Mean;
import ru.kuptservol.jml.v2.SGD;

/**
 * @author Sergey Kuptsov
 */
public class SGDTest {

    @Test
    public void test() throws IOException {
        DataSet mnist = DataSets.MNIST(Paths.get("/opt/jml/mnist"));

        Tensor x = Tensor.tensor(mnist.train.x);
        Tensor y = Tensor.tensor(mnist.train.y);

        Model model = new SimpleModel();

        SGD sgd = new SGD(x, y, model);

        sgd.fit();
    }

    private class SimpleModel implements Model {

        private Layer[] layers;

        public SimpleModel() {
            this.layers = new Layer[]{
                    new Linear(Init.Kaiming.init(784, 50), Init.Kaiming.init(50)),
                    new Relu0Mean(),
                    new Linear(Init.Kaiming.init(50, 10), Init.Kaiming.init(10))};
        }

        @Override
        public Tensor forward(Tensor x) {
            for (Layer layer : layers) {
                x = layer.forward(x);
            }

            return x;
        }

        @Override
        public void backward() {
            for (int i = layers.length - 1; i >= 0; i--) {
                layers[i].backward();
            }
        }

        @Override
        public Layer[] getLayers() {
            return layers;
        }
    }
}
