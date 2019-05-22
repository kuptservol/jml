package ru.kuptservol.jml.result.function;

import ru.kuptservol.jml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class OutputFunctions {

    public final static OutputFunction MaxIndex = M::maxIndex;

    public final static OutputFunction MaxVal = M::maxVal;
}
