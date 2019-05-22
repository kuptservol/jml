package ru.kuptservol.jml.metric.result;

/**
 * @author Sergey Kuptsov
 */
public class ResultHandlers {

    public final static ResultHandler LOG = new LogResultHandler();

    public final static ResultHandler Empty =
            new ResultHandler() {
                @Override
                public ResultHandler wrap(double cost, String dataLabel, String format) {
                    return this;
                }
            };

    public static ResultHandler Graph(PlotGraphResultHandler graph) {
        return (cost, dataLabel, format) -> {
            graph.addPoint(cost, dataLabel);
            return graph;
        };
    }

    public static ResultHandler GraphAndLog(PlotGraphResultHandler graph) {
        return (cost, dataLabel, format) -> {
            graph.addPoint(cost, dataLabel);
            graph.print();
            return new LogResultHandler(cost, dataLabel, format);
        };
    }
}
