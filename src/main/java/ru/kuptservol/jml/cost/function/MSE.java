package ru.kuptservol.jml.cost.function;

import lombok.AllArgsConstructor;
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
@AllArgsConstructor
public class MSE implements CostFunction {

    @Builder.Default
    private ResultHandler resultHandler = ResultHandlers.LOG;

    @Override
    public double cost(Model m, double[][] X, double[][] Y) {
        double cost = 0;

        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.resultF.process(m.forward(X[i])) - m.resultF.process(Y[i]), 2) / X.length;
        }

        cost += m.regularization.addCost(m, X.length);

        return cost;
    }

    @Override
    public double[] backprop(Layer lastLayer, double[] activations, double[] y) {
        return M.hadamartR(M.minusR(activations, y), lastLayer.dActDZ());
    }

    @Override
    public String printFormat() {
        return "mse: %.3f";
    }
}
