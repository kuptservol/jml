package ru.kuptservol.ml.matrix;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import static java.lang.Double.POSITIVE_INFINITY;

/**
 * @author Sergey Kuptsov
 */
public class M {

    public static double[][] dotR(double[][] a, double[][] b) {
        int aRows = a.length;

        checkArraySize(aRows);
        checkArraySize(b.length);
        checkSameLength(a[0].length, b.length);

        int n = b.length;

        int bColumns = b[0].length;
        double[][] result = new double[aRows][bColumns];

        for (int lrow = 0; lrow < aRows; lrow++) {
            for (int rcol = 0; rcol < bColumns; rcol++) {

                double resVal = 0;
                for (int i = 0; i < n; i++) {
                    resVal += a[lrow][i] * b[i][rcol];
                }

                result[lrow][rcol] = resVal;
            }
        }

        return result;
    }

    public static double[][] dotR(double[] a, double[] b) {
        int aRows = a.length;
        int bCols = b.length;

        checkArraySize(aRows);
        checkArraySize(bCols);

        double[][] result = new double[aRows][bCols];

        for (int lrow = 0; lrow < aRows; lrow++) {
            for (int rcol = 0; rcol < bCols; rcol++) {

                result[lrow][rcol] = a[lrow] * b[rcol];
            }
        }

        return result;
    }

    public static double[] dotR(double[][] a, double[] b) {
        int aRows = a.length;

        checkArraySize(aRows);
        checkArraySize(b.length);
        checkSameLength(a[0].length, b.length);

        int bColumns = b.length;
        double[] result = new double[aRows];

        for (int lrow = 0; lrow < aRows; lrow++) {
            double resVal = 0;

            for (int rcol = 0; rcol < bColumns; rcol++) {
                resVal += a[lrow][rcol] * b[rcol];
            }

            result[lrow] = resVal;
        }

        return result;
    }

    public static double[] dotR(double[] a, double[][] b) {

        checkArraySize(a.length);
        checkArraySize(b.length);
        checkSameLength(a.length, b.length);

        int bColumns = b[0].length;
        int bRows = b.length;
        double[] result = new double[bColumns];

        for (int rCol = 0; rCol < bColumns; rCol++) {
            double resVal = 0;

            for (int bRow = 0; bRow < bRows; bRow++) {
                resVal += a[bRow] * b[bRow][rCol];
            }

            result[rCol] = resVal;
        }

        return result;
    }

    public static void plus(double[][] a, double[][] b) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);
        checkSameLength(a[0].length, b[0].length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] += b[i][j];
            }
        }
    }

    public static void plus(double[] a, double[] b) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);

        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    public static void minus(double[] a, double[] b) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);

        for (int i = 0; i < a.length; i++) {
            a[i] = a[i] - b[i];
        }
    }

    public static double[] minusR(double[] a, double b) {
        checkArraySize(a.length);

        double[] result = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b;
        }

        return result;
    }

    public static double[] minusR(double[] a, double[] b) {
        checkArraySize(a.length);

        double[] result = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }

        return result;
    }

    public static double ln(double a) {
        return Math.log1p(a);
    }

    public static double nanToNum(double a) {
        if (Double.isFinite(a)) {
            return a;
        }

        if (Double.isNaN(a)) {
            return 0;
        }

        if (a == POSITIVE_INFINITY) {
            return Double.MAX_VALUE;
        }


        return Double.MIN_VALUE;
    }

    public static double[] plusR(double[] a, double[] b) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);

        double[] result = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }

        return result;
    }

    public static double[][] FR(Function<Double, Double> f, double[][] z) {
        double[][] a = new double[z.length][z[0].length];
        for (int i = 0; i < z.length; i++) {
            for (int j = 0; j < z[0].length; j++) {
                a[i][j] = f.apply(z[i][j]);
            }
        }

        return a;
    }

    public static double[] FR(Function<Double, Double> f, double[] z) {
        double[] a = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            a[i] = f.apply(z[i]);
        }

        return a;
    }

    public static void F(double[] a, double[] b, BiFunction<Double, Double, Double> bFunc) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);

        for (int i = 0; i < a.length; i++) {
            a[i] = bFunc.apply(a[i], b[i]);
        }
    }

    public static double[] FR(double[] a, double[] b, BiFunction<Double, Double, Double> bFunc) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);

        double[] result = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            result[i] = bFunc.apply(a[i], b[i]);
        }

        return result;
    }

    public static void F(double[][] a, double[][] b, BiFunction<Double, Double, Double> bFunc) {
        checkArraySize(a.length);
        checkArraySize(b.length);

        checkSameLength(a.length, b.length);
        checkSameLength(a[0].length, b[0].length);

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] = bFunc.apply(a[i][j], b[i][j]);
            }
        }
    }

    public static double[][] T(double[][] a) {
        checkArraySize(a.length);

        double[][] result = new double[a[0].length][a.length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[j][i] = a[i][j];
            }
        }

        return result;
    }

    public static void shuffle(double[][] a) {
        int index;
        double[] temp;
        Random random = new Random();
        for (int i = a.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            temp = a[index];
            a[index] = a[i];
            a[i] = temp;
        }
    }

    public static void shuffle(double[][] x, double[][] y) {
        checkSameLength(x.length, y.length);

        int index;
        double[] tempX;
        double[] tempY;
        Random random = new Random();
        for (int i = x.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            tempX = x[index];
            tempY = y[index];
            x[index] = x[i];
            y[index] = y[i];
            x[i] = tempX;
            y[i] = tempY;
        }
    }

    private static void checkSameLength(int a, int b) {
        if (a != b) {
            throw new IllegalArgumentException("Expected same length - passed {" + a + "} and {" + b + "}");
        }
    }

    private static void checkArraySize(int a) {
        if (a <= 0) {
            throw new IllegalArgumentException("Empty array not expected");
        }
    }

    public static class Tuple2<T1, T2> {
        public T1 left;
        public T2 right;

        public static <T1, T2> Tuple2<T1, T2> tuple(T1 one, T2 two) {
            return new Tuple2<>(one, two);
        }

        public Tuple2(T1 left, T2 right) {
            this.left = left;
            this.right = right;
        }
    }

    public static Tuple2<Data, Data> split(double[][] x, double[][] y, int length) {
        checkArraySize(x.length);
        checkArraySize(y.length);
        checkSameLength(x.length, y.length);
        checkSameLength(x.length, y.length);

        if (length <= 0 || length >= x.length) {
            throw new IllegalArgumentException("Length must be in (0;length)");
        }

        double[][] xLeft = new double[length][x[0].length];
        double[][] xRight = new double[x.length - length][x[0].length];
        double[][] yLeft = new double[length][y[0].length];
        double[][] yRight = new double[x.length - length][y[0].length];

        for (int i = 0; i < x.length; i++) {
            if (i < length) {
                xLeft[i] = x[i];
                yLeft[i] = y[i];
            } else {
                xRight[i - length] = x[i];
                yRight[i - length] = y[i];
            }
        }

        return Tuple2.tuple(Data.cons(xLeft, yLeft), Data.cons(xRight, yRight));
    }

    public static Tuple2<Data, Data> split(int[][] x, int[][] y, int length) {
        checkArraySize(x.length);
        checkArraySize(y.length);
        checkSameLength(x.length, y.length);
        checkSameLength(x.length, y.length);

        if (length <= 0 || length >= x.length) {
            throw new IllegalArgumentException("Length must be in (0;length)");
        }

        int[][] xLeft = new int[length][x[0].length];
        int[][] xRight = new int[x.length - length][x[0].length];
        int[][] yLeft = new int[length][y[0].length];
        int[][] yRight = new int[x.length - length][y[0].length];

        for (int i = 0; i < x.length; i++) {
            if (i < length) {
                xLeft[i] = x[i];
                yLeft[i] = y[i];
            } else {
                xRight[i - length] = x[i];
                yRight[i - length] = y[i];
            }
        }

        return Tuple2.tuple(Data.cons(xLeft, yLeft), Data.cons(xRight, yRight));
    }

    public static Data[] chunk(double[][] x, double[][] y, int chunkSize) {
        checkArraySize(x.length);
        checkArraySize(y.length);
        checkSameLength(x.length, y.length);

        int numOfChunks = (int) Math.ceil((double) y.length / chunkSize);
        Data[] output = new Data[numOfChunks];

        for (int i = 0; i < numOfChunks; ++i) {
            int start = i * chunkSize;
            int length = Math.min(y.length - start, chunkSize);

            double[][] tempX = new double[length][x[0].length];
            double[][] tempY = new double[length][y[0].length];
            System.arraycopy(y, start, tempY, 0, length);
            System.arraycopy(x, start, tempX, 0, length);
            output[i] = Data.cons(tempX, tempY);
        }

        return output;
    }

    public static double maxIndex(double[] values) {
        double maxI = -1;
        Double maxV = null;

        for (int i = 0; i < values.length; i++) {
            if (maxV == null) {
                maxV = values[i];
                maxI = 0;
            } else {
                if (values[i] > maxV) {
                    maxV = values[i];
                    maxI = i;
                }
            }
        }

        return maxI;
    }


    public static String asPixels(double[][] pixels) {
        StringBuilder sb = new StringBuilder();

        for (int row = 0; row < pixels.length; row++) {
            sb.append("|");
            for (int col = 0; col < pixels[row].length; col++) {
                double pixelVal = pixels[row][col];
                if (pixelVal == 0)
                    sb.append(" ");
                else if (pixelVal < 256 / 3.0)
                    sb.append(".");
                else if (pixelVal < 2 * (256 / 3.0))
                    sb.append("x");
                else
                    sb.append("X");
            }
            sb.append("|\n");
        }

        return sb.toString();
    }

    public static double[][] to(double[] a, int row, int col) {
        checkArraySize(a.length);
        checkSameLength(a.length, row * col);

        double[][] result = new double[row][col];

        int r = 0;
        int c = 0;
        for (int i = 0; i < a.length; i++) {
            result[r][c] = a[i];

            if (c == col - 1) {
                r++;
                c = 0;
            } else {
                c++;
            }
        }

        return result;
    }

    public static double maxVal(double[] values) {
        checkArraySize(values.length);
        Double maxV = null;

        for (int i = 0; i < values.length; i++) {
            if (maxV == null) {
                maxV = values[i];
            } else {
                if (values[i] > maxV) {
                    maxV = values[i];
                }
            }
        }

        return maxV;
    }

    public static class Data {
        public double[][] x;
        public double[][] y;
        public int size;

        public static Data cons(double[][] x, double[][] y) {
            checkSameLength(x.length, y.length);
            Data data = new Data();
            data.x = x;
            data.y = y;
            data.size = x.length;

            return data;
        }

        public static Data cons(int[][] x, int[][] y) {
            checkSameLength(x.length, y.length);
            Data data = new Data();
            double[][] xD = new double[x.length][x[0].length];
            double[][] yD = new double[y.length][y[0].length];

            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++) {
                    xD[i][j] = x[i][j];
                }
            }

            for (int i = 0; i < y.length; i++) {
                for (int j = 0; j < y[0].length; j++) {
                    yD[i][j] = y[i][j];
                }
            }

            data.x = xD;
            data.y = yD;
            data.size = x.length;

            return data;
        }

    }

    public static String toString(double[][] matrix) {
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                s.append(matrix[i][j]).append(" ");
            }
            s.append(System.lineSeparator());
        }

        return s.toString();
    }

    public static double[] hadamartR(double[] a, double[] b) {
        checkSameLength(a.length, b.length);

        double[] res = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            res[i] = a[i] * b[i];
        }

        return res;
    }
}
