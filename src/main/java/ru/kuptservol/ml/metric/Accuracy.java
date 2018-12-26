package ru.kuptservol.ml.metric;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.ml.metric.result.Metric;
import ru.kuptservol.ml.metric.result.Metrics;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class Accuracy implements ru.kuptservol.ml.metric.Metric {

    @Builder.Default
    private Metric metric = Metrics.LOG;

    @Override
    public Metric execute(Model m, double[][] X, double[][] Y) {
        double length = X.length;
        double correct = 0;

        for (int i = 0; i < length; i++) {
            if (m.resultF.process(m.forward(X[i])) == m.resultF.process(Y[i]))
                correct++;
        }

        return metric.create(100 * correct / length, "Accuracy: %.3f %%");
    }
}
