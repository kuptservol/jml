package ru.kuptservol.jml.metric;

import lombok.AllArgsConstructor;
import lombok.Builder;
import ru.kuptservol.jml.model.Model;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class Accuracy implements Metric {

    @Override
    public double execute(Model m, double[][] X, double[][] Y) {
        double length = X.length;
        double correct = 0;

        for (int i = 0; i < length; i++) {
            if (m.resultF.process(m.forward(X[i])) == m.resultF.process(Y[i]))
                correct++;
        }

        return 100 * correct / length;
    }

    @Override
    public String printFormat() {
        return "accuracy %.3f%%";
    }
}
