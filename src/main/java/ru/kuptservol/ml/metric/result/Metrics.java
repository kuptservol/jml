package ru.kuptservol.ml.metric.result;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class Metrics {

    public final static Metric LOG = LogMetric::new;

    public static Metric GRAPH(PlotGraphMetric graph) {
        return (cost, pattern) -> {
            graph.addPoint(cost);
            return graph;
        };
    }

    public static Metric GRAPH_AND_LOG(PlotGraphMetric graph) {
        return (cost, pattern) -> {
            graph.addPoint(cost);
            graph.print();
            return new LogMetric(cost, pattern);
        };
    }
}
