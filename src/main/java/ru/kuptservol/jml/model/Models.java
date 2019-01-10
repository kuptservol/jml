package ru.kuptservol.jml.model;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import ru.kuptservol.jml.layer.Layers;

/**
 * @author Sergey Kuptsov
 */
public class Models {

    public static Model.ModelBuilder linear(Double learningRate, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .linear(learningRate, 1, size)
                .build();

        modelBuilder.layers(layers);

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, Double dropout, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .linear(learningRate, dropout, size)
                .build();

        modelBuilder.layers(layers);

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .linear(0.5, 1, size)
                .build();

        modelBuilder.layers(layers);

        return modelBuilder;
    }

    public static Model load(Path from) throws IOException, ClassNotFoundException {
        try (ObjectInputStream objectInputStream = new ObjectInputStream(Files.newInputStream(from))) {
            return (Model) objectInputStream.readObject();
        }
    }

    public static void save(Path to, Model model) throws IOException {
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(Files.newOutputStream(to))) {
            objectOutputStream.writeObject(model);
        }
    }
}
