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
public class L2Regularization implements Regularization {

    /**
     * regularization parameter
     */
    @Builder.Default
    private double lambda = 0;

    @Override
    public double norm(double[][] weights) {
        return M.l1Norm(weights);
    }

    @Override
    public double addCost(Model m, int trainLength) {
        return (lambda / trainLength) * norm(m.layers);
    }

    @Override
    public double reg(double learningRate, int batchSize, double w) {
        return w * learningRate * (lambda / batchSize);
    }
}
