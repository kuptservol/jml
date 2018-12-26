package ru.kuptservol.ml.cost.function;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.metric.result.Metric;
import ru.kuptservol.ml.metric.result.Metrics;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class MSE implements CostFunction {

    @Builder.Default
    private Metric metric = Metrics.LOG;

    @Override
    public Metric cost(Model m, double[][] X, double[][] Y) {
        double cost = 0;

        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.resultF.process(m.forward(X[i])) - m.resultF.process(Y[i]), 2) / X.length;
        }

        return metric.create(cost, "MSE: %.3f");
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
