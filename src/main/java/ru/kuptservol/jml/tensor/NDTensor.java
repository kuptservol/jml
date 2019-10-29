package ru.kuptservol.jml.tensor;

import java.util.Collections;
import java.util.Objects;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Sergey Kuptsov
 * Tensor powered by nd4j matrix operations
 */
public class NDTensor {

    private final SDVariable matrix;
    private final SameDiff computeGraph;
    private final boolean requiresGrad;

    private NDTensor(SDVariable matrix) {
        this.matrix = matrix;
        this.computeGraph = matrix.getSameDiff();
        this.requiresGrad = true;
    }

    private NDTensor(INDArray array) {
        this(array, true);
    }

    private NDTensor(INDArray array, boolean requiresGrad) {
        SameDiff computeGraph = SameDiff.create();
        this.computeGraph = computeGraph;
        if (requiresGrad) {
            this.matrix = computeGraph.var(array);
            this.requiresGrad = true;
        } else {
            this.matrix = computeGraph.constant(array);
            this.requiresGrad = false;
        }
    }

    public static NDTensor tensor(double[][] matrix) {
        return new NDTensor(Nd4j.create(matrix));
    }

    public static NDTensor tensor(int... shape) {
        return new NDTensor(Nd4j.create(shape));
    }

    public static NDTensor tensor(double[] vector) {
        return new NDTensor(Nd4j.create(vector));
    }

    public static NDTensor tensor(double scalar) {
        SameDiff computeGraph = SameDiff.create();
        return new NDTensor(computeGraph.constant(scalar));
    }

    public static NDTensor randn(int... shape) {
        return new NDTensor(Nd4j.randn(shape));
    }

    public NDTensor mmul(NDTensor t) {
        return new NDTensor(matrix.mmul(t.matrix));
    }

    public SDVariable getMatrix() {
        return matrix;
    }

    //    public NDTensor exp() {
//        return matrix.pow()
//        return new NDTensor(MatrixFunctions.exp(matrix.dup()));
//    }

    public NDTensor mul(NDTensor t) {
        return new NDTensor(matrix.mul(t.matrix));
    }

    public NDTensor mul(double val) {
        return new NDTensor(matrix.mul(val));
    }

    public NDTensor normalize(double mean, double std) {
        return new NDTensor(matrix.subi(mean).divi(std));
    }

    public NDTensor minus(double value) {
        return new NDTensor(matrix.sub(value));
    }

    public NDTensor minus(NDTensor t) {
        return new NDTensor(matrix.sub(t.matrix));
    }

    public void backward() {
        computeGraph.execBackwards(Collections.emptyMap());
    }
//
//    public NDTensor broadcast(NDTensor to) {
//        if (to.matrix.columns > matrix.columns && to.matrix.rows == matrix.rows) {
//            DoubleMatrix broadcasted = new DoubleMatrix(matrix.rows, to.matrix.columns);
//            for (int row = 0; row < matrix.rows; row++) {
//                int brColumnI = 0;
//                for (int column = 0; column < to.matrix.columns; column++) {
//                    broadcasted.put(row, column, matrix.get(row, brColumnI));
//                    if (brColumnI == matrix.columns - 1) {
//                        brColumnI = 0;
//                    } else {
//                        brColumnI++;
//                    }
//                }
//            }
//
//            return new NDTensor(broadcasted);
//        }
//
//        return this;
//    }

    public NDTensor pow(double grade) {
        return new NDTensor(matrix.pow(grade));
    }

    public INDArray grad() {
        return matrix.gradient().eval();
    }

//    public double meanD() {
//        return matrix.mean();
//    }

    public NDTensor mean() {
        return new NDTensor(matrix.mean());
    }

    public INDArray val() {
       return matrix.eval();
    }

//    public NDTensor sum() {
//        return NDTensor.tensor(matrix.sum());
//    }

//    public NDTensor sum(int dim) {
//        if (dim == 0) {
//            return sum();
//        } else if (dim == 1) {
//            DoubleMatrix maxByRows = new DoubleMatrix(matrix.rows, 1);
//
//            for (int i = 0; i < matrix.rows; i++) {
//                maxByRows.put(i, 0, matrix.getRow(i).sum());
//            }
//
//            return new NDTensor(maxByRows);
//        } else {
//            throw new NotImplementedException();
//        }
//    }
/*
    public NDTensor log() {
        return new NDTensor(MatrixFunctions.log(matrix));
        return computeGraph.cnn.avgPooling3d()
    }

    public NDTensor neg() {
        return new NDTensor(matrix.neg());
    }

    public NDTensor max() {
        return NDTensor.tensor(matrix.max());
    }

    public NDTensor max(int dim) {
        if (dim == 0) {
            return NDTensor.tensor(matrix.max());
        } else if (dim == 1) {
            DoubleMatrix maxByRows = new DoubleMatrix(matrix.rows, 1);

            for (int i = 0; i < matrix.rows; i++) {
                maxByRows.put(i, 0, matrix.getRow(i).max());
            }

            return new NDTensor(maxByRows);
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

    public NDTensor clamp_min(double val) {
        DoubleMatrix matrix_new = matrix.dup();
        for (int i = 0; i < matrix_new.rows; i++) {
            for (int j = 0; j < matrix_new.columns; j++) {
                if (matrix_new.get(i, j) < val) {
                    matrix_new.put(i, j, 0);
                }
            }
        }

        return new NDTensor(matrix_new);
    }

    public NDTensor minusi(NDTensor val) {
        matrix.subi(val.matrix);
        return this;
    }

    @Override
    public String toString() {
        return matrix.toString("%.4f", "", "", " ", "\n");
    }

    public NDTensor F(Function<Double, Double> func) {
        DoubleMatrix matrix_new = matrix.dup();
        for (int i = 0; i < matrix_new.rows; i++) {
            for (int j = 0; j < matrix_new.columns; j++) {
                matrix_new.put(i, j, func.apply(matrix_new.get(i, j)));
            }
        }

        return new NDTensor(matrix_new);
    }

    public NDTensor plus(NDTensor t) {
        return new NDTensor(matrix.add(t.matrix));
    }

    public NDTensor T() {
        return new NDTensor(matrix.transpose());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NDTensor)) return false;
        NDTensor tensor = (NDTensor) o;
        return Objects.equals(matrix, tensor.matrix) &&
                Objects.equals(grad, tensor.grad);
    }

    public NDTensor div(double val) {
        return new NDTensor(matrix.div(val));
    }

    public NDTensor div(NDTensor val) {
        return new NDTensor(matrix.div(val.matrix));
    }


 */

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NDTensor)) return false;
        NDTensor ndTensor = (NDTensor) o;
        return Objects.equals(matrix, ndTensor.matrix) &&
                Objects.equals(computeGraph, ndTensor.computeGraph);
    }

    @Override
    public int hashCode() {
        return Objects.hash(matrix, computeGraph);
    }
}
