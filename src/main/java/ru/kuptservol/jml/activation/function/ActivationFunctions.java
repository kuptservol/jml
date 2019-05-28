package ru.kuptservol.jml.activation.function;

/**
 * @author Sergey Kuptsov
 */
public class ActivationFunctions {

    public final static Sigmoid Sigmoid = new Sigmoid();

    public final static Tanh Tanh = new Tanh();

    public final static ReLU ReLU = new ReLU();

    public final static Softmax Softmax = new Softmax();

    public final static StableSoftmax StableSoftmax = new StableSoftmax();
}
