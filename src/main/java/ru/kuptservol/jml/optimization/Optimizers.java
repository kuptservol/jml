package ru.kuptservol.jml.optimization;

/**
 * @author Sergey Kuptsov
 */
public class Optimizers {

    public static Optimizer RMS_PROP(double momentumCoeff) {
        return new RMSprop(momentumCoeff);
    }

    public static Optimizer NONE() {
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
