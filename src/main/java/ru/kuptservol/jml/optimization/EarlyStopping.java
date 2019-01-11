package ru.kuptservol.jml.optimization;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class EarlyStopping {
    private final static Logger logger = LoggerFactory.getLogger(EarlyStopping.class);

    private double[] series;
    private boolean full;
    private double maxValue;
    private int p = 0;

    public EarlyStopping(int numOfSeries) {
        this.series = new double[numOfSeries];
    }

    public void countSerie(double value) {
        if (p == series.length) {
            p = 0;
            full = true;
        }

        series[p] = value;
        if (value > maxValue) {
            maxValue = value;
        }

        p++;
    }

    public boolean doContinue() {
        if (!full) return true;

        for (double seriesVal : series) {
            if (seriesVal >= maxValue) {
                return true;
            }
        }

        logger.info("Stopping cause no improvement for {} series more than maxValue = {} was found", series.length, maxValue);
        return false;
    }
}
