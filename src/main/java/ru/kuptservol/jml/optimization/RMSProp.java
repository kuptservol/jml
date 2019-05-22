package ru.kuptservol.jml.optimization;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class RMSProp implements Optimizer {

    private double[][] squaredBatchWeightGrads;
    private double[] squaredBatchBiasGrads;
    // beta
    private final double momentumCoeff;
    private final double epsilon = 1E-8;

    public RMSProp(double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
    }

    private RMSProp(double momentumCoeff, int in, int out) {
        this.momentumCoeff = momentumCoeff;
        this.squaredBatchWeightGrads = new double[in][out];
        this.squaredBatchBiasGrads = new double[out];
    }

    @Override
    public Optimizer init(int in, int out) {
        return new RMSProp(momentumCoeff, in, out);
    }

    @Override
    public void optimize(double[][] batchWeightGrads) {
        M.F(squaredBatchWeightGrads, batchWeightGrads,
                (squaredBatchWeightGrad, batchWeightGrad)
                        -> squaredBatchWeightGrad * momentumCoeff + Math.pow(batchWeightGrad, 2) * (1 - momentumCoeff));

        M.F(batchWeightGrads, squaredBatchWeightGrads,
                (batchWeightGrad, squaredBatchWeightGradsPrev) ->
                        M.nanToNum(batchWeightGrad / (Math.sqrt(squaredBatchWeightGradsPrev) + epsilon)));
    }

    @Override
    public void optimize(double[] batchBiasGrads) {
        M.F(squaredBatchBiasGrads, batchBiasGrads,
                (squaredBatchBiasGrad, batchBiasGrad)
                        -> squaredBatchBiasGrad * momentumCoeff + Math.pow(batchBiasGrad, 2) * (1 - momentumCoeff));

        M.F(batchBiasGrads, squaredBatchBiasGrads,
                (batchBiasGrad, squaredBatchBiasGradsPrevTmp)
                        -> M.nanToNum(batchBiasGrad / (Math.sqrt(squaredBatchBiasGradsPrevTmp) + epsilon)));
    }
}
