package ru.kuptservol.ml.cost.function;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.metric.result.MetricsResult;
import ru.kuptservol.ml.metric.result.MetricsResults;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class MSE implements CostFunction {

    @Builder.Default
    private MetricsResult metricsResult = MetricsResults.LOG;

    @Override
    public MetricsResult execute(Model m, double[][] X, double[][] Y) {
        double cost = 0;

        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.evaluate(X[i]) - m.resultFunction.apply(Y[i]), 2) / X.length;
        }

        return metricsResult.create(cost, "MSE: %.3f");
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
