package ru.kuptservol.jml.cost.function;

import lombok.Builder;
import ru.kuptservol.jml.layer.Layer;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.metric.result.ResultHandler;
import ru.kuptservol.jml.metric.result.ResultHandlers;
import ru.kuptservol.jml.model.Model;

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
            double[] Y_hat = m.forward(X[i]);

            double costPart = 0;
            for (int j = 0; j < Y[i].length; j++) {
                costPart -= M.nanToNum(Y[i][j] * Math.log(Y_hat[j]));
            }

            cost += costPart / X.length;
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
