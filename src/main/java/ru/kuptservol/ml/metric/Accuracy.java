package ru.kuptservol.ml.metric;

import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public class Accuracy implements Metric {

    @Override
    public MetricsResult execute(Model m, double[][] X, double[][] Y) {
        double length = X.length;
        double correct = 0;

        for (int i = 0; i < length; i++) {
            if (m.evaluate(X[i]) == m.resultFunction.apply(Y[i]))
                correct++;
        }

        return new SimpleMetricsResult(100 * correct / length, "accuracy %.3f %%");
    }
}
