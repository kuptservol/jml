package ru.kuptservol.jml.optimization;

/**
 * @author Sergey Kuptsov
 */
public class Optimizations {
    public static L2Regularization L2_REG(double lambda) {
        return L2Regularization.builder().lambda(lambda).build();
    }

    public static L1Regularization L1_REG(double lambda) {
        return L1Regularization.builder().lambda(lambda).build();
    }

    public static Dropout DROPOUT(double perc) {
        return new Dropout(perc);
    }

    public static EarlyStopping EARLY_STOPPING(int noImprovementForSeriesLen) {
        return new EarlyStopping(noImprovementForSeriesLen);
    }
}
