package ru.kuptservol.jml.cost.function;

import lombok.Builder;
import ru.kuptservol.jml.layer.Layer;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.metric.result.ResultHandlers;
import ru.kuptservol.jml.model.Model;

import static ru.kuptservol.jml.matrix.M.ln;

/**
 * @author Sergey Kuptsov
 */
@Builder
public class CrossEntropy implements CostFunction {

    @Builder.Default
    private ResultHandler resultHandler = ResultHandlers.LOG;

    @Override
    public double cost(Model m, double[][] X, double[][] Y) {
        double cost = 0;

        for (int i = 0; i < X.length; i++) {
            double a = m.resultF.process(m.forward(X[i]));
            double y = m.resultF.process(Y[i]);

            cost += M.nanToNum(-y * ln(a) - (1 - y) * ln(1 - a)) / X.length;
        }

        cost += m.regularization.addCost(m, X.length);

        return cost;
    }

    @Override
    //todo: check if act function is sigmoid
    public double[] backprop(Layer layer, double[] a, double[] y) {
        return M.minusR(a, y);
    }

    @Override
    public String printFormat() {
        return "cross_entropy: %.3f";
    }
}
