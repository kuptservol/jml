package ru.kuptservol.jml.train;

import java.io.Serializable;
import java.util.Optional;

import ru.kuptservol.jml.data.DataSet;
import ru.kuptservol.jml.matrix.M;
import ru.kuptservol.jml.model.Model;

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
