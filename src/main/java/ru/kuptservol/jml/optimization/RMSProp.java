package ru.kuptservol.jml.optimization;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class Momentum implements Optimizer {

    private double[][] weightBatchGradsPrev;
    private double[] biasBatchGradsPrev;
    // beta
    private final double momentumCoeff;

    public Momentum(double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
    }

    private Momentum(double momentumCoeff, int in, int out) {
        this.momentumCoeff = momentumCoeff;
        this.weightBatchGradsPrev = new double[in][out];
        this.biasBatchGradsPrev = new double[out];
    }

    @Override
    public Optimizer init(int in, int out) {
        return new Momentum(momentumCoeff, in, out);
    }

    @Override
    public void optimize(double[][] weightBatchGrads) {
        M.F(weightBatchGrads, weightBatchGradsPrev,
                (weightGrads, prevWeightGrads) -> prevWeightGrads * momentumCoeff + weightGrads * (1 - momentumCoeff));

        weightBatchGradsPrev = weightBatchGrads;
    }

    @Override
    public void optimize(double[] biasBatchGrads) {
        M.F(biasBatchGrads, biasBatchGradsPrev,
                (biasGrads, prevBiasGrads) -> prevBiasGrads * momentumCoeff + biasGrads * (1 - momentumCoeff));

        biasBatchGradsPrev = biasBatchGrads;
    }
}
