package ru.kuptservol.ml.metric;

import lombok.AllArgsConstructor;
import lombok.Builder;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class SimpleMetricsResult implements MetricsResult {
    public double value;
    public String pattern;

    @Override
    public String print() {
        return String.format(pattern, value);
    }
}
