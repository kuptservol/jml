package ru.kuptservol.ml.metric.result;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class MetricsResults {

    public final static MetricsResult LOG = LogMetricsResult::new;

    public static MetricsResult GRAPH(PlotGraphMetricsResult graph) {
        return (cost, pattern) -> {
            graph.addPoint(cost);
            return graph;
        };
    }

    public static MetricsResult GRAPH_AND_LOG(PlotGraphMetricsResult graph) {
        return (cost, pattern) -> {
            graph.addPoint(cost);
            graph.print();
            return new LogMetricsResult(cost, pattern);
        };
    }
}
