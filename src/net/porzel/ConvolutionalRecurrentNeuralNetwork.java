package net.porzel;

public class ConvolutionalRecurrentNeuralNetwork extends ConvolutionalNeuralNetwork {
    /**
     * Constructs a neural network with the specified layer sizes.
     *
     * @param layers an array of integers specifying the number of neurons in each layer. Must have at least 3 layers.
     * @throws RuntimeException if the number of layers is less than 3 or if any layer has fewer than 1 neuron
     */
    public ConvolutionalRecurrentNeuralNetwork(int[] layers) {
        super(layers);
    }


}
