package ru.kuptservol.ml.metric.result;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
public class PlotGraphMetricsResult implements MetricsResult {
    private final static Logger logger = LoggerFactory.getLogger(PlotGraphMetricsResult.class);

    private boolean initialized = false;

    private XYChart chart;

    private final List<Double> points = new ArrayList<>();
    private SwingWrapper<XYChart> chartSwingWrapper;

    public void addPoint(double p) {
        points.add(p);
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
            initialized = true;
        }

        javax.swing.SwingUtilities.invokeLater(() -> {
            chart.updateXYSeries("cost", null, points, null);
            chartSwingWrapper.repaintChart();
        });

        return "";
    }

    @Override
    public MetricsResult create(double cost, String pattern) {
        return new PlotGraphMetricsResult();
    }

    private void initialize() {
        chart = new XYChartBuilder().width(600).height(400).title("Cost").xAxisTitle("Epoch").yAxisTitle("Cost").build();

        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.addSeries("cost", new double[]{0}, new double[]{0});

        chartSwingWrapper = new SwingWrapper<>(chart);
        chartSwingWrapper.displayChart();
    }
}
