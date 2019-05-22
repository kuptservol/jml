package ru.kuptservol.jml.optimization;

import java.util.function.Function;

import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.weight.initializer.WeightInitializer;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
public class RMSProp implements Optimizer {

    private double[][] squaredBatchWeightGrads;
    private double[] squaredBatchBiasGrads;
    // beta
    private final double momentumCoeff;

    public RMSProp(double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
    }

    private RMSProp(double momentumCoeff, int in, int out) {
        this.momentumCoeff = momentumCoeff;
        // random for start prev
        WeightInitializer weightInitializer = WeightInitializers.SharpGaussian;
        this.squaredBatchWeightGrads = weightInitializer.initWeights(in, out);
        this.squaredBatchBiasGrads = weightInitializer.initBiases(out);
    }

    @Override
    public Optimizer init(int in, int out) {
        return new RMSProp(momentumCoeff, in, out);
    }

    @Override
    public void optimize(double[][] batchWeightGrads) {
        double[][] squaredBatchWeightGradsPrevs = M.FR(Function.identity(), squaredBatchWeightGrads);

        M.F(squaredBatchWeightGrads, batchWeightGrads,
                (squaredBatchWeightGrad, batchWeightGrad)
                        -> squaredBatchWeightGrad * momentumCoeff + Math.pow(batchWeightGrad, 2) * (1 - momentumCoeff));

        M.F(batchWeightGrads, squaredBatchWeightGradsPrevs,
                (batchWeightGrad, squaredBatchWeightGradsPrev) ->
                        M.nanToNum(batchWeightGrad / Math.sqrt(squaredBatchWeightGradsPrev)));

    }

    @Override
    public void optimize(double[] batchBiasGrads) {
        double[] squaredBatchBiasGradsPrev = M.FR(Function.identity(), squaredBatchBiasGrads);

        M.F(squaredBatchBiasGrads, batchBiasGrads,
                (squaredBatchBiasGrad, batchBiasGrad)
                        -> squaredBatchBiasGrad * momentumCoeff + Math.pow(batchBiasGrad, 2) * (1 - momentumCoeff));

        M.F(batchBiasGrads, squaredBatchBiasGradsPrev,
                (batchBiasGrad, squaredBatchBiasGradsPrevTmp) -> M.nanToNum(batchBiasGrad / Math.sqrt(squaredBatchBiasGradsPrevTmp)));
    }
}
