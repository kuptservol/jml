package ru.kuptservol.ml.metric.result;

import lombok.AllArgsConstructor;

/**
 * @author Sergey Kuptsov
 */
@AllArgsConstructor
public class LogMetric implements Metric {
    public double value;
    public String pattern;

    @Override
    public String print() {
        return String.format(pattern, value);
    }

    public Metric create(double cost, String pattern) {
        return new LogMetric(cost, pattern);
    }
}
