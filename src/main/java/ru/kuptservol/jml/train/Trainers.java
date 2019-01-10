package ru.kuptservol.jml.train;

/**
 * @author Sergey Kuptsov
 */
public class Trainers {

    public static SGD.SGDBuilder SGD(int batchSize, int epochs) {
        return new SGD.SGDBuilder()
                .batchSize(batchSize)
                .epochs(epochs);
    }

    public static SGD.SGDBuilder SGD() {
        return new SGD.SGDBuilder();
    }
}
