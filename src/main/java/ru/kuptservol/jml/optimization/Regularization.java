package ru.kuptservol.jml.optimization;

import ru.kuptservol.jml.layer.Layers;
import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface Regularization {
    double norm(double[][] weights);

    double addCost(Model m, int trainLength);

    double reg(double learningRate, int batchSize, double weight);

    default double norm(Layers layers) {
        double[] norm = new double[1];

        layers.forEach(layer -> norm[0] += layer.norm(this));

        return norm[0];
    }
}
