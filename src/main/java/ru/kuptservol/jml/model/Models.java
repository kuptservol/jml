package ru.kuptservol.jml.model;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import ru.kuptservol.jml.activation.function.ActivationFunction;
import ru.kuptservol.jml.activation.function.ActivationFunctions;
import ru.kuptservol.jml.layer.Layers;
import ru.kuptservol.jml.optimization.Optimizations;
import ru.kuptservol.jml.optimization.Optimizer;
import ru.kuptservol.jml.optimization.Optimizers;
import ru.kuptservol.jml.weight.initializer.WeightInitializer;
import ru.kuptservol.jml.weight.initializer.WeightInitializers;

/**
 * @author Sergey Kuptsov
 */
@AllArgsConstructor
@Getter
public class Models {

    @Builder
    public static class LinearModelBuilder {
        @Builder.Default
        public double learningRate = 0.01;
        @Builder.Default
        public ActivationFunction activationFunction = ActivationFunctions.Sigmoid;
        @Builder.Default
        public WeightInitializer weightInitializer = WeightInitializers.Gaussian(1);
        @Builder.Default
        public double dropout = 0;
        @Builder.Default
        public Optimizer optimizer = Optimizers.None();
    }

    public static Model.ModelBuilder linear(LinearModelBuilder linearModelBuilder, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(
                        linearModelBuilder.dropout,
                        linearModelBuilder.weightInitializer,
                        linearModelBuilder.activationFunction,
                        linearModelBuilder.optimizer,
                        size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(linearModelBuilder.learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, WeightInitializers.Gaussian(1), ActivationFunctions.Sigmoid, Optimizers.Momentum(0), size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, WeightInitializer weightInitializer, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, weightInitializer, ActivationFunctions.Sigmoid, Optimizers.Momentum(0), size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, WeightInitializer weightInitializer, Double momentumCoEff, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0,
                        weightInitializer,
                        ActivationFunctions.Sigmoid,
                        Optimizers.Momentum(momentumCoEff),
                        size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(learningRate));

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
                .fullyConnected(0, weightInitializer, activationFunction, Optimizers.Momentum(momentumCoEff), size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Double learningRate, Double dropout, Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(dropout, WeightInitializers.Gaussian(1), ActivationFunctions.Sigmoid, Optimizers.Momentum(0), size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(learningRate));

        return modelBuilder;
    }

    public static Model.ModelBuilder linear(Integer... size) {
        Model.ModelBuilder modelBuilder = new Model.ModelBuilder();

        Layers layers = Layers
                .fullyConnected(0, WeightInitializers.Gaussian(1), ActivationFunctions.Sigmoid, Optimizers.Momentum(0), size)
                .build();

        modelBuilder.layers(layers);
        modelBuilder.adaptiveLearningRate(Optimizations.ConstLearningRate(0.5));

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
