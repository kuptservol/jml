package ru.kuptservol.jml.tensor;

import java.util.Objects;
import java.util.function.Function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Sergey Kuptsov
 * Tensor powered by jblas matrix operations
 * todo: support backprop
 */
public class Tensor {

    public final DoubleMatrix matrix;
    public final int rank;
    public final int[] shape;
    public Tensor grad;

    private Tensor(DoubleMatrix matrix) {
        this.matrix = matrix;

        if (matrix.rows == 1 && matrix.columns == 1) {
            this.rank = 0;
            this.shape = new int[]{1};
        } else if (matrix.rows == 1 || matrix.columns == 1) {
            this.rank = 1;
            this.shape = new int[]{Math.max(matrix.rows, matrix.columns)};
        } else {
            this.rank = 2;
            this.shape = new int[]{matrix.rows, matrix.columns};
        }
    }

    public Tensor getRowRange(int from, int to) {
        return new Tensor(matrix.getRange(from, to, 0, matrix.columns));
    }

    public static Tensor tensor(double[][] matrix) {
        return new Tensor(new DoubleMatrix(matrix));
    }

    public static Tensor tensor(int... shape) {
        return new Tensor(new DoubleMatrix(shape[0], shape[1]));
    }

    public static Tensor tensor(double[] vector) {
        return new Tensor(new DoubleMatrix(1, vector.length, vector));
    }

    public static Tensor tensor(double scalar) {
        return new Tensor(new DoubleMatrix(1, 1, scalar));
    }

    public static Tensor randn(int... shape) {
        if (shape.length == 1) {
            return new Tensor(DoubleMatrix.randn(1, shape[0]));
        } else {
            return new Tensor(DoubleMatrix.randn(shape[0], shape[1]));
        }
    }

    public Tensor mmul(Tensor t) {
        return new Tensor(matrix.mmul(t.matrix));
    }

    public Tensor exp() {
        return new Tensor(MatrixFunctions.exp(matrix.dup()));
    }

    public Tensor mul(Tensor t) {
        return new Tensor(matrix.mul(t.matrix));
    }

    public Tensor mul(double val) {
        return new Tensor(matrix.mul(val));
    }

    public Tensor normalize(double mean, double std) {
        return new Tensor(matrix.subi(mean).divi(std));
    }

    public Tensor minus(double value) {
        return new Tensor(matrix.sub(value));
    }

    public Tensor minus(Tensor t) {
        return new Tensor(matrix.sub(t.matrix));
    }

    public Tensor broadcast(Tensor to) {
        if (to.matrix.columns > matrix.columns && to.matrix.rows == matrix.rows) {
            DoubleMatrix broadcasted = new DoubleMatrix(matrix.rows, to.matrix.columns);
            for (int row = 0; row < matrix.rows; row++) {
                int brColumnI = 0;
                for (int column = 0; column < to.matrix.columns; column++) {
                    broadcasted.put(row, column, matrix.get(row, brColumnI));
                    if (brColumnI == matrix.columns - 1) {
                        brColumnI = 0;
                    } else {
                        brColumnI++;
                    }
                }
            }

            return new Tensor(broadcasted);
        }

        return this;
    }

    public Tensor pow(double grade) {
        return new Tensor(MatrixFunctions.pow(matrix, grade));
    }

    public double meanD() {
        return matrix.mean();
    }

    public Tensor mean() {
        return Tensor.tensor(matrix.mean());
    }

    public Tensor sum() {
        return Tensor.tensor(matrix.sum());
    }

    public Tensor sum(int dim) {
        if (dim == 0) {
            return sum();
        } else if (dim == 1) {
            DoubleMatrix maxByRows = new DoubleMatrix(matrix.rows, 1);

            for (int i = 0; i < matrix.rows; i++) {
                maxByRows.put(i, 0, matrix.getRow(i).sum());
            }

            return new Tensor(maxByRows);
        } else {
            throw new NotImplementedException();
        }
    }

    public Tensor log() {
        return new Tensor(MatrixFunctions.log(matrix));
    }

    public Tensor neg() {
        return new Tensor(matrix.neg());
    }

    public Tensor max() {
        return Tensor.tensor(matrix.max());
    }

    public Tensor max(int dim) {
        if (dim == 0) {
            return Tensor.tensor(matrix.max());
        } else if (dim == 1) {
            DoubleMatrix maxByRows = new DoubleMatrix(matrix.rows, 1);

            for (int i = 0; i < matrix.rows; i++) {
                maxByRows.put(i, 0, matrix.getRow(i).max());
            }

            return new Tensor(maxByRows);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    public double sumD() {
        return matrix.sum();
    }

    public double std() {
        double mean = meanD();
        double std = 0;

        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.columns; j++) {
                double diff = matrix.get(i, j) - mean;
                std += diff * diff;
            }
        }

        return Math.sqrt(std / (matrix.rows * matrix.columns));
    }

    public Tensor clamp_min(double val) {
        DoubleMatrix matrix_new = matrix.dup();
        for (int i = 0; i < matrix_new.rows; i++) {
            for (int j = 0; j < matrix_new.columns; j++) {
                if (matrix_new.get(i, j) < val) {
                    matrix_new.put(i, j, 0);
                }
            }
        }

        return new Tensor(matrix_new);
    }

    public Tensor minusi(Tensor val) {
        matrix.subi(val.matrix);
        return this;
    }

    @Override
    public String toString() {
        return matrix.toString("%.4f", "", "", " ", "\n");
    }

    public Tensor F(Function<Double, Double> func) {
        DoubleMatrix matrix_new = matrix.dup();
        for (int i = 0; i < matrix_new.rows; i++) {
            for (int j = 0; j < matrix_new.columns; j++) {
                matrix_new.put(i, j, func.apply(matrix_new.get(i, j)));
            }
        }

        return new Tensor(matrix_new);
    }

    public Tensor plus(Tensor t) {
        return new Tensor(matrix.add(t.matrix));
    }

    public Tensor T() {
        return new Tensor(matrix.transpose());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;
        Tensor tensor = (Tensor) o;
        return Objects.equals(matrix, tensor.matrix) &&
                Objects.equals(grad, tensor.grad);
    }

    public Tensor div(double val) {
        return new Tensor(matrix.div(val));
    }

    public Tensor div(Tensor val) {
        return new Tensor(matrix.div(val.matrix));
    }
}
