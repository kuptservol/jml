package ru.kuptservol.ml.result.function;

import ru.kuptservol.ml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class OutputFunctions {

    public final static OutputFunction MAX_INDEX = M::maxIndex;

    public final static OutputFunction MAX_VAL = M::maxVal;
}
