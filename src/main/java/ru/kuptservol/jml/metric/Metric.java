package ru.kuptservol.jml.metric;

import java.io.Serializable;

import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface Metric extends Serializable {

    double execute(Model m, double[][] X, double[][] Y);

    String printFormat();
}
