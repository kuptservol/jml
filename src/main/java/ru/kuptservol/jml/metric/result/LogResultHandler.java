package ru.kuptservol.jml.metric.result;

import lombok.AllArgsConstructor;

/**
 * @author Sergey Kuptsov
 */
@AllArgsConstructor
public class LogResultHandler implements ResultHandler {
    public double value;
    public String dataLabel;
    public String pattern;

    public LogResultHandler() {
    }

    @Override
    public String print() {
        return "[" + dataLabel + " " + String.format(pattern, value) + "]";
    }

    @Override
    public ResultHandler wrap(double cost, String dataLabel, String format) {
        return new LogResultHandler(cost, dataLabel, format);
    }
}
