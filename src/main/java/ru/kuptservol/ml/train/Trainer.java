package ru.kuptservol.ml.train;

import java.io.Serializable;
import java.util.Optional;

import ru.kuptservol.ml.data.DataSet;
import ru.kuptservol.ml.matrix.M;
import ru.kuptservol.ml.model.Model;

/**
 * @author Sergey Kuptsov
 */
public interface Trainer extends Serializable {
    default void train(Model m, double[][] trainX, double[][] trainY) {
        train(m, DataSet
                .builder()
                .train(M.Data.cons(trainX, trainY))
                .build());
    }

    default void train(Model m, double[][] trainX, double[][] trainY, double[][] testX, double[][] testY) {
        train(m, DataSet
                .builder()
                .train(M.Data.cons(trainX, trainY))
                .test(Optional.of(M.Data.cons(testX, testY)))
                .build());
    }

    void train(Model m, DataSet dataSet);
}
