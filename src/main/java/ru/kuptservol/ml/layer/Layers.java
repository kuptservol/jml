package ru.kuptservol.ml.layer;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.function.Consumer;

import lombok.AllArgsConstructor;
import lombok.Builder;

/**
 * @author Sergey Kuptsov
 */
@Builder
@AllArgsConstructor
public class Layers implements Serializable {

    private LinkedList<Layer> hiddenLayers;

    public static LayersBuilder linear(double learningRate, Integer... size) {
        if (size.length < 2) {
            throw new IllegalArgumentException("Size cannot be less than 2");
        }

        LayersBuilder builder = Layers.builder();

        LinkedList<Layer> hiddenLayers = new LinkedList<>();

        LinearLayer first = LinearLayer.builder()
                .in(size[0])
                .out(size[1])
                .learningRate(learningRate)
                .build();

        hiddenLayers.addLast(first);

        for (int i = 2; i < size.length; i++) {
            LinearLayer nextL = LinearLayer.builder().in(size[i - 1]).out(size[i]).build();
            hiddenLayers.addLast(nextL);
        }

        return builder.hiddenLayers(hiddenLayers);
    }

    public double[] forward(double[] input) {
        double[] activations = input;

        for (Layer next : hiddenLayers) {
            activations = next.forward(activations);
        }

        return activations;
    }

    public double[] backprop(double[] dCostDaNextLayer) {
        double[] dCostDaNext = dCostDaNextLayer;

        Iterator<Layer> layerIterator = hiddenLayers.descendingIterator();
        while (layerIterator.hasNext()) {
            Layer next = layerIterator.next();
            dCostDaNext = next.backprop(dCostDaNext);
        }

        return dCostDaNext;
    }

    public void forEach(Consumer<Layer> consumer) {
        hiddenLayers.forEach(consumer);
    }
}
