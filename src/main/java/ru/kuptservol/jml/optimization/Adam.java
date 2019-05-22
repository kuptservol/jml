package ru.kuptservol.jml.optimization;

import java.util.function.Function;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class Adam implements Optimizer {

    private double[][] squaredBatchWeightGrads;
    private double[] squaredBatchBiasGrads;
    private double[][] weightBatchGradsPrev;
    private double[] biasBatchGradsPrev;
    // beta
    private final double momentumCoeff;
    private final double momentumSqrCoeff;
    private final double epsilon = 1E-8;

    public Adam(double coeff, double sqrCoeff) {
        this.momentumCoeff = coeff;
        this.momentumSqrCoeff = sqrCoeff;
    }

    private Adam(double momentumCoeff, double sqrCoeff, int in, int out) {
        this.momentumCoeff = momentumCoeff;
        this.momentumSqrCoeff = sqrCoeff;
        // random for start prev
        this.squaredBatchWeightGrads = new double[in][out];
        this.squaredBatchBiasGrads = new double[out];
        this.weightBatchGradsPrev = new double[in][out];
        this.biasBatchGradsPrev = new double[out];
    }

    @Override
    public Optimizer init(int in, int out) {
        return new Adam(momentumCoeff, momentumSqrCoeff, in, out);
    }

    @Override
    public void optimize(double[][] weightBatchGrads) {
        M.F(squaredBatchWeightGrads, weightBatchGrads,
                (squaredBatchWeightGrad, batchWeightGrad)
                        -> squaredBatchWeightGrad * momentumSqrCoeff + Math.pow(batchWeightGrad, 2) * (1 - momentumSqrCoeff));

        M.F(weightBatchGrads, weightBatchGradsPrev,
                (weightGrads, prevWeightGrads) -> prevWeightGrads * momentumCoeff + weightGrads * (1 - momentumCoeff));

        weightBatchGradsPrev = M.FR(Function.identity(), weightBatchGrads);

        M.F(weightBatchGrads, squaredBatchWeightGrads,
                (batchWeightGrad, squaredBatchWeightGradsPrev) ->
                        M.nanToNum(batchWeightGrad / (Math.sqrt(squaredBatchWeightGradsPrev) + epsilon)));
    }

    @Override
    public void optimize(double[] biasBatchGrads) {
        M.F(squaredBatchBiasGrads, biasBatchGrads,
                (squaredBatchBiasGrad, batchBiasGrad)
                        -> squaredBatchBiasGrad * momentumSqrCoeff + Math.pow(batchBiasGrad, 2) * (1 - momentumSqrCoeff));

        M.F(biasBatchGrads, biasBatchGradsPrev,
                (biasGrads, prevBiasGrads) -> prevBiasGrads * momentumCoeff + biasGrads * (1 - momentumCoeff));

        biasBatchGradsPrev = M.FR(Function.identity(), biasBatchGrads);

        M.F(biasBatchGrads, squaredBatchBiasGrads,
                (batchBiasGrad, squaredBatchBiasGradsPrevTmp)
                        -> M.nanToNum(batchBiasGrad / (Math.sqrt(squaredBatchBiasGradsPrevTmp) + epsilon)));
    }
}
