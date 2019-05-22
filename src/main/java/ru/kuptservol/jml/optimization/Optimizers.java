package ru.kuptservol.jml.optimization;

/**
 * @author Sergey Kuptsov
 */
public class Optimizers {

    public static Optimizer Momentum(double momentumCoeff) {
        return new Momentum(momentumCoeff);
    }

    public static Optimizer RMSProp(double coeff) {
        return new RMSProp(coeff);
    }

    public static Optimizer Adam(double coeff, double sqrCoeff) {
        return new Adam(coeff, sqrCoeff);
    }

    public static Optimizer None() {
        return new Optimizer() {
            @Override
            public Optimizer init(int in, int out) {
                return this;
            }

            @Override
            public void optimize(double[][] weightBatchGrads) {

            }

            @Override
            public void optimize(double[] biasBatchGrads) {

            }
        };
    }
}
