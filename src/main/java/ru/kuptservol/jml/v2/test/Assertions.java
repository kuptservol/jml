package ru.kuptservol.jml.v2.test;

import ru.kuptservol.jml.tensor.Tensor;

/**
 * @author Sergey Kuptsov
 */
public class Assertions {

    public static void assertNear(Tensor one, Tensor another, double tol) {
        if (!one.matrix.sameSize(another.matrix)) {
            throw new AssertionError("Different size");
        }


        for (int i = 0; i < one.matrix.rows; i++) {
            for (int j = 0; j < one.matrix.columns; j++) {
                if (Math.abs(one.matrix.get(i, j) - another.matrix.get(i, j)) > tol) {
                    throw new AssertionError("Differs " + one.matrix.get(i, j) + " and " + another.matrix.get(i, j));
                }
            }
        }
    }

    public static void assertNear(double one, double two, double tol) {
        if (Math.abs(one - two) > tol) {
            throw new AssertionError("Differs " + one + " and " + two);
        }
    }
}
