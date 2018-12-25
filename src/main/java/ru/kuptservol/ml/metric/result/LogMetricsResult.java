package ru.kuptservol.ml.metric.result;

import lombok.AllArgsConstructor;

/**
 * @author Sergey Kuptsov
 */
@AllArgsConstructor
public class LogMetricsResult implements MetricsResult {
    public double value;
    public String pattern;

    @Override
    public String print() {
        return String.format(pattern, value);
    }

    public MetricsResult create(double cost, String pattern) {
        return new LogMetricsResult(cost, pattern);
    }
}
