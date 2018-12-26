package ru.kuptservol.ml.cost.function;

/**
 * @author Sergey Kuptsov
 */
public class CostFunctions {

    public static MSE.MSEBuilder MSE = new MSE.MSEBuilder();

    public static CrossEntropy.CrossEntropyBuilder CROSS_ENTROPY = new CrossEntropy.CrossEntropyBuilder();
}
