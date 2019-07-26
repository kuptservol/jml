package ru.kuptservol.jml.matrix;

import java.util.Arrays;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Sergey Kuptsov
 */
public class MTest {

    @Test
    public void dot() {
        double[][] a = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
        };

        double[] b = {1, 1, 1};

        double[] c = {6, 15, 24, 33};

        assertEquals(Arrays.toString(M.dotR(a, b)), Arrays.toString(c));
    }

    @Test
    public void mean() {
        double[][] a = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };

        assertEquals(M.mean(a), 5.0, 0.0);
    }

    @Test
    public void std() {
        double[][] a = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };

        assertEquals(7.74, M.std(a), 0.01);
    }

    @Test
    public void normalize() {
        double[][] a = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };

        a = M.normalizeR(a);

        assertTrue(1 - M.std(a) < 1);
        assertTrue(M.mean(a) < 1);
    }

    @Test
    public void dot2() {
        double[] a = {1, 2, 3};

        double[] b = {4, 5, 6, 7};

        double[][] c = {
                {4, 5, 6, 7},
                {8, 10, 12, 14},
                {12, 15, 18, 21}
        };

        assertEquals(M.toString(M.dotR(a, b)), M.toString(c));
    }

    @Test
    public void shuffle() {
        double[][] a = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9},
                {4, 11, 12},
        };

        double[][] b = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9},
                {4, 11, 12},
        };

        M.shuffle(a, b);

        assertEquals(b[0][0], a[0][0]);
        assertEquals(b[1][0], a[1][0]);
        assertEquals(b[2][0], a[2][0]);
        assertEquals(b[3][0], a[3][0]);
    }

    @Test
    public void to() {
        double[] x = {1, 2, 3, 4};

        double[][] result = {{1, 2}, {3, 4}};

        assertEquals(M.toString(result), M.toString(M.to(x, 2, 2)));
    }

    @Test
    public void asPixels() {
        double[][] pxls = {{0, 2}, {0, 4}};

        System.out.println(M.asPixels(pxls));
    }

    @Test
    public void chunk() {
        double[][] x = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9}
        };

        double[][] y = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };

        M.Data[] chunk = M.chunk(x, y, 2);

        double[][] x0 = {
                {1, 2, 3},
                {2, 5, 6}
        };
        double[][] x1 = {
                {3, 8, 9}
        };
        double[][] y0 = {
                {1, 2, 3},
                {4, 5, 6}
        };
        double[][] y1 = {
                {7, 8, 9}
        };

        assertEquals(chunk[0].x, x0);
        assertEquals(chunk[1].x, x1);
        assertEquals(chunk[0].y, y0);
        assertEquals(chunk[1].y, y1);
    }

    @Test
    public void f() {
        double[][] x = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9}
        };

        double[][] y = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9}
        };

        double[][] result = {
                {3, 5, 7},
                {5, 11, 13},
                {7, 17, 19}
        };

        M.F(x, y, (xE, yE) -> xE + yE + 1);
        assertEquals(M.toString(x), M.toString(result));
        assertEquals(M.toString(y), M.toString(y));
    }

    @Test
    public void split() {
        double[][] x = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9}
        };

        double[][] y = {
                {1, 2, 3},
                {2, 5, 6},
                {3, 8, 9}
        };

        M.Tuple2<M.Data, M.Data> split = M.split(x, y, 1);
        assertEquals(split.left.x.length, 1);
        assertEquals(split.right.x.length, 2);
        assertEquals(split.left.y.length, 1);
        assertEquals(split.right.y.length, 2);

        M.Tuple2<M.Data, M.Data> split2 = M.split(x, y, 2);
        assertEquals(split2.left.x.length, 2);
        assertEquals(split2.right.x.length, 1);
        assertEquals(split2.left.y.length, 2);
        assertEquals(split2.right.y.length, 1);
    }

    @Test
    public void plus() {
        double[] a = {5, 14, 23, 32};

        double[] b = {1, 1, 1, 1};

        double[] c = {6, 15, 24, 33};

        assertEquals(Arrays.toString(M.plusR(a, b)), Arrays.toString(c));
    }
}
