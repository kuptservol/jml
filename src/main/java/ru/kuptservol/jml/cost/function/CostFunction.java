package ru.kuptservol.jml.cost.function;

import java.io.Serializable;

import ru.kuptservol.jml.layer.Layer;
import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface CostFunction extends Serializable {

    double cost(Model m, double[][] trainX, double[][] trainY);

    /**
     * return Dcost/dA*activationFunction'
     */
    double[] backprop(Layer lastLayer, double[] activations, double[] y);

    String printFormat();
}
