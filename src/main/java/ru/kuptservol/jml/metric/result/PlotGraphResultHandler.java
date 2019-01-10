package ru.kuptservol.jml.metric.result;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Sergey Kuptsov
 */
public class PlotGraphResultHandler implements ResultHandler {
    private final static Logger logger = LoggerFactory.getLogger(PlotGraphResultHandler.class);

    private boolean initialized = false;
    private Optional<Path> graphPathFileName = Optional.empty();

    public PlotGraphResultHandler() {

    }

    public static PlotGraphResultHandler cons(Path graphPathFileName) {
        return new PlotGraphResultHandler(graphPathFileName);
    }

    public PlotGraphResultHandler(Path graphPathFileName) {
        this.graphPathFileName = Optional.of(graphPathFileName);
    }

    private XYChart chart;

    private final Map<String, List<Double>> points = new HashMap<>();
    private SwingWrapper<XYChart> chartSwingWrapper;

    public void addPoint(double p, String dataLabel) {
        points.compute(dataLabel, (s, points) -> {
            if (points == null) {
                List<Double> l = new ArrayList<>();
                l.add(p);
                return l;
            } else {
                points.add(p);
                return points;
            }
        });
    }

    public void save(Path graphPathFileName) {
        try {
            BitmapEncoder.saveBitmapWithDPI(chart, graphPathFileName.toString(), BitmapEncoder.BitmapFormat.PNG, 300);
        } catch (IOException e) {
            logger.error("Saving graph to file error", e);
        }
    }

    @Override
    public String print() {
        if (!initialized) {
            initialize();
            graphPathFileName.ifPresent(graphPathFileNamePath ->
                    Runtime.getRuntime().addShutdownHook(new Thread(() -> save(graphPathFileNamePath))));
            initialized = true;
        }

        javax.swing.SwingUtilities.invokeLater(() -> {
            points.forEach((key, value) -> chart.updateXYSeries(key, null, value, null));
            chartSwingWrapper.repaintChart();
        });

        return "";
    }

    @Override
    public ResultHandler wrap(double cost, String dataLabel, String format) {
        return new PlotGraphResultHandler();
    }

    private void initialize() {
        chart = new XYChartBuilder().width(600).height(400).title("Cost").xAxisTitle("Epoch").yAxisTitle("Cost").build();

        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        chart.addSeries("train", new double[]{0}, new double[]{0});
        chart.addSeries("test", new double[]{0}, new double[]{0});

        chartSwingWrapper = new SwingWrapper<>(chart);
        chartSwingWrapper.displayChart();
    }
}
