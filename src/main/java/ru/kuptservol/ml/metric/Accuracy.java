package ru.kuptservol.ml.metric;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.ml.metric.result.MetricsResult;
import ru.kuptservol.ml.metric.result.MetricsResults;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class Accuracy implements Metric {

    @Builder.Default
    private MetricsResult metricsResult = MetricsResults.LOG;

    @Override
    public MetricsResult execute(Model m, double[][] X, double[][] Y) {
        double length = X.length;
        double correct = 0;

        for (int i = 0; i < length; i++) {
            if (m.evaluate(X[i]) == m.resultFunction.apply(Y[i]))
                correct++;
        }

        return metricsResult.create(100 * correct / length, "Accuracy: %.3f %%");
    }
}
