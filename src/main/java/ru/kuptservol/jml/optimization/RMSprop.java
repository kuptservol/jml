package ru.kuptservol.jml.optimization;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class RMSprop implements Optimizer {

    private double[][] weightBatchGradsPrev;
    private double[] biasBatchGradsPrev;
    // beta
    private final double momentumCoeff;

    public RMSprop(double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
    }

    private RMSprop(double momentumCoeff, int in, int out) {
        this.momentumCoeff = momentumCoeff;
        this.weightBatchGradsPrev = new double[in][out];
        this.biasBatchGradsPrev = new double[out];
    }

    @Override
    public Optimizer init(int in, int out) {
        return new RMSprop(momentumCoeff, in, out);
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
