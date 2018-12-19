package ru.kuptservol.ml.result.function;

import ru.kuptservol.ml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class ResultFunctions {

    public final static ResultFunction MAX_INDEX = M::maxIndex;

    public final static ResultFunction MAX_VAL = M::maxVal;
}
