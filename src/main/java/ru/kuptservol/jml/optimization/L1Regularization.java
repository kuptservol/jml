package ru.kuptservol.jml.optimization;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class L1Regularization implements Regularization {

    /**
     * regularization parameter
     */
    @Builder.Default
    private double lambda = 0;

    @Override
    public double norm(double[][] weights) {
        return Math.pow(M.l2Norm(weights), 2);
    }

    @Override
    public double addCost(Model m, int trainLength) {
        return 0.5 * (lambda / trainLength) * norm(m.layers);
    }

    @Override
    public double reg(double learningRate, int batchSize, double w) {
        return learningRate * (lambda / batchSize) * (w < 0 ? -1 : 1);
    }
}
