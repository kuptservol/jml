package ru.kuptservol.jml.optimization;

/**
 * @author Sergey Kuptsov
 */
public class Optimizations {
    public static L2Regularization L2Reg(double lambda) {
        return L2Regularization.builder().lambda(lambda).build();
    }

    public static L1Regularization L1Reg(double lambda) {
        return L1Regularization.builder().lambda(lambda).build();
    }

    public static Dropout Dropout(double perc) {
        return new Dropout(perc);
    }

    public static EarlyStopping EarlyStopping(int noImprovementForSeriesLen) {
        return new EarlyStopping(noImprovementForSeriesLen);
    }

    public static AdaptiveLearningRate ConstLearningRate(double learningRate) {
        return new ConstAdaptiveLearningRate(learningRate);
    }

    public static AdaptiveLearningRate ConstDecreasingLearningRate(double learningRate) {
        return new ConstDecreasingAdaptiveLearningRate(learningRate);
    }
}
