package ru.kuptservol.jml.model;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import ru.kuptservol.jml.activation.function.ActivationFunction;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.layer.Layers;
import ru.kuptservol.jml.optimization.Optimizations;
import ru.kuptservol.jml.weight.initializer.WeightInitializer;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
public class Models {

    public static Model.ModelBuilder linear(Double learningRate, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, WeightInitializers.GAUSSIAN(1), ActivationFunctions.SIGMOID, 0, size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.CONST_LEARNING_RATE(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, WeightInitializer weightInitializer, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, weightInitializer, ActivationFunctions.SIGMOID, 0, size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.CONST_LEARNING_RATE(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, WeightInitializer weightInitializer, Double momentumCoEff, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, weightInitializer, ActivationFunctions.SIGMOID, momentumCoEff, size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.CONST_LEARNING_RATE(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(
            Double learningRate,
            ActivationFunction activationFunction,
            WeightInitializer weightInitializer,
            Double momentumCoEff,
            Integer... size)
    {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, weightInitializer, activationFunction, momentumCoEff, size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.CONST_LEARNING_RATE(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, Double dropout, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(dropout, WeightInitializers.GAUSSIAN(1), ActivationFunctions.SIGMOID, 0, size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.CONST_LEARNING_RATE(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, WeightInitializers.GAUSSIAN(1), ActivationFunctions.SIGMOID, 0, size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.CONST_LEARNING_RATE(0.5));

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
