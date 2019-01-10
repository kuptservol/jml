package ru.kuptservol.jml.test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ru.kuptservol.jml.metric.Metrics;
import ru.kuptservol.jml.model.Model;
import ru.kuptservol.jml.model.Models;
import ru.kuptservol.jml.result.function.OutputFunctions;
import org.junit.Test;

/**
 * @author Sergey Kuptsov
 */
public class ModelTest {

    @Test
    public void saveAndLoad() throws IOException, ClassNotFoundException {
        Model model = Models.linear(0.1, 784, 30, 10)
                .resultF(OutputFunctions.MAX_INDEX)
                .metrics(Metrics.ACCURACY.build())
                .build();

        Path path = Paths.get("/tmp/model1");
        model.save(path);

        Model modelLoaded = Models.load(path);
    }
}
