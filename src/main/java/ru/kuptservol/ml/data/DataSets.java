package ru.kuptservol.ml.data;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.kuptservol.ml.data.mnist.MnistReader;
import ru.kuptservol.ml.matrix.M;

/**
 * @author Sergey Kuptsov
 */
public class DataSets {
    private final static Logger logger = LoggerFactory.getLogger(DataSets.class);

    public static DataSet MNIST(Path pathToDownload) throws IOException {
        logger.debug("Loading MNIST dataset");

        FilesLoader.downloadIfNotExists(pathToDownload, "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-imgs.gz");
        FilesLoader.gunzipIfNotExists(pathToDownload.toString(), "train-imgs.gz", "train-imgs");
        FilesLoader.downloadIfNotExists(pathToDownload, "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels.gz");
        FilesLoader.gunzipIfNotExists(pathToDownload.toString(), "train-labels.gz", "train-labels");
        FilesLoader.downloadIfNotExists(pathToDownload, "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "test-imgs.gz");
        FilesLoader.gunzipIfNotExists(pathToDownload.toString(), "test-imgs.gz", "test-imgs");
        FilesLoader.downloadIfNotExists(pathToDownload, "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "test-labels.gz");
        FilesLoader.gunzipIfNotExists(pathToDownload.toString(), "test-labels.gz", "test-labels");

        int[][] trainImgs = MnistReader.getImages(Paths.get(pathToDownload.toString(), "train-imgs").toString());
        int[][] trainLabels = MnistReader.getLabels(Paths.get(pathToDownload.toString(), "train-labels").toString());

        M.Tuple2<M.Data, M.Data> valSplit = M.split(trainImgs, trainLabels, (int) (trainImgs.length * 0.9));

        int[][] testImgs = MnistReader.getImages(Paths.get(pathToDownload.toString(), "test-imgs").toString());
        int[][] testLabels = MnistReader.getLabels(Paths.get(pathToDownload.toString(), "test-labels").toString());

        DataSet dataSet = DataSet.builder()
                .train(valSplit.left)
                .validation(Optional.of(valSplit.right))
                .test(Optional.of((M.Data.cons(testImgs, testLabels))))
                .build();

        logger.debug("Loading MNIST dataset finished");

        return dataSet;
    }
}
