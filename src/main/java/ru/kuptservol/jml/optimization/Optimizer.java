package ru.kuptservol.jml.optimization;

/**
 * @author Sergey Kuptsov
 */
public interface Optimizer {
    Optimizer init(int in, int out);

    void optimize(double[][] weightBatchGrads);

    void optimize(double[] biasBatchGrads);
}
