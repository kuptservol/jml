package ru.kuptservol.jml.metric.result;

/**
 * @author Sergey Kuptsov
 */
public class ResultHandlers {

    public final static ResultHandler LOG = new LogResultHandler();

    public final static ResultHandler EMPTY =
            new ResultHandler() {
                @Override
                public ResultHandler wrap(double cost, String dataLabel, String format) {
                    return this;
                }
            };

    public static ResultHandler GRAPH(PlotGraphResultHandler graph) {
        return (cost, dataLabel, format) -> {
            graph.addPoint(cost, dataLabel);
            return graph;
        };
    }

    public static ResultHandler GRAPH_AND_LOG(PlotGraphResultHandler graph) {
        return (cost, dataLabel, format) -> {
            graph.addPoint(cost, dataLabel);
            graph.print();
            return new LogResultHandler(cost, dataLabel, format);
        };
    }
}
