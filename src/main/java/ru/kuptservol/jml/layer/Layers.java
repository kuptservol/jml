package ru.kuptservol.jml.layer;

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

    public Layer last() {
        return hiddenLayers.getLast();
    }

    public static LayersBuilder linear(double learningRate, double dropout, Integer... size) {
        if (size.length < 2) {
            throw new IllegalArgumentException("Size cannot be less than 2");
        }

        LayersBuilder builder = Layers.builder();

        LinkedList<Layer> hiddenLayers = new LinkedList<>();

        LinearLayer first = LinearLayer.builder()
                .in(size[0])
                .out(size[1])
                .dropout(dropout)
                .learningRate(learningRate)
                .build();

        hiddenLayers.addLast(first);

        for (int i = 2; i < size.length; i++) {
            LinearLayer.LinearLayerBuilder nextLB = LinearLayer.builder()
                    .in(size[i - 1])
                    .out(size[i])
                    .learningRate(learningRate);

            if (i < size.length - 1) {
                nextLB.dropout(dropout);
            }
            hiddenLayers.addLast(nextLB.build());
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
        boolean lastLayer = true;

        Iterator<Layer> layerIterator = hiddenLayers.descendingIterator();
        while (layerIterator.hasNext()) {
            Layer next = layerIterator.next();
            dCostDaNext = lastLayer ? next.lastLayerBackprop(dCostDaNext) : next.backprop(dCostDaNext);
            lastLayer = false;
        }

        return dCostDaNext;
    }

    public void forEach(Consumer<Layer> consumer) {
        hiddenLayers.forEach(consumer);
    }
}
