package ru.kuptservol.ml.cost.function;

import lombok.Builder;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.metric.result.Metric;
import ru.kuptservol.ml.metric.result.Metrics;
import ru.kuptservol.ml.model.Model;

import static ru.kuptservol.ml.matrix.M.ln;
import static ru.kuptservol.ml.matrix.M.nanToNum;

/**
 * @author Sergey Kuptsov <kuptservol@yandex-team.ru>
 */
@Builder
public class CrossEntropy implements CostFunction {

    @Builder.Default
    private Metric metrics = Metrics.LOG;

    @Override
    public Metric cost(Model m, double[][] X, double[][] Y) {
        double cost = 0;

        for (int i = 0; i < X.length; i++) {
            double a = m.resultF.process(m.forward(X[i]));
            double y = m.resultF.process(Y[i]);

            cost += nanToNum(-y * ln(a) - (1 - y) * ln(1 - a)) / X.length;
        }

        return metrics.create(cost, "Cross entropy: %.3f");
    }

    @Override
    public double[] backprop(double[] A, double[] Y) {
        return M.FR(A, Y,
                (a, y) -> nanToNum(-y / a - (1 - y) / (1 - a))
        );
    }
}
