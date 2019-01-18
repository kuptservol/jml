package ru.kuptservol.jml.optimization;

import java.util.Random;

/**
 * @author Sergey Kuptsov
 */
public class Dropout {

    private double perc;

    private final Random random = new Random();

    private double[] mask;

    public Dropout(double perc) {
        this.perc = perc;
    }

    public void initMask(int length) {
        mask = new double[length];

        double scale = perc == 1 ? 0 : 1 / (1 - perc);
        for (int i = 0; i < mask.length; i++) {
            mask[i] = perc == 0 ? 1 : Math.pow(random.nextDouble(), 2) >= perc ? scale : 0;
        }
    }

    public double[] mask() {
        return mask;
    }
}
