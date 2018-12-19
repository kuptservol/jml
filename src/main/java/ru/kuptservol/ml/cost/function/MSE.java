package ru.kuptservol.ml.cost.function;

import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.metric.MetricsResult;
import ru.kuptservol.ml.metric.SimpleMetricsResult;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public class MSE implements CostFunction {

    @Override
    public MetricsResult execute(Model m, double[][] X, double[][] Y) {
        double cost = 0;

        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.evaluate(X[i]) - m.resultFunction.apply(Y[i]), 2) / X.length;
        }

        return new SimpleMetricsResult(cost, "MSE: %.3f");
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
