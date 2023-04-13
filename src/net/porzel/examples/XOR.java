package net.porzel.examples;

import net.porzel.ConvolutionalNeuralNetwork;
import net.porzel.functions.ActivationFunction;
import net.porzel.functions.CostFunction;
import net.porzel.functions.LossFunction;
import net.porzel.functions.WeightInitialization;

import java.io.File;

public class XOR {
    public static void main(String[] args) {
        ConvolutionalNeuralNetwork neuralNetwork = new ConvolutionalNeuralNetwork(new int[] {2, 32, 16, 8, 4, 1})
                .setLearningRate(0.025)
                .setActivationFunction(ActivationFunction.SIGMOID())
                .setWeightInitializationFunction(WeightInitialization.LECUN())
                .setLossFunction(LossFunction.BINARY_CROSS_ENTROPY_LOSS())
                .setCostFunction(CostFunction.MEDIAN());


        double[][] input = new double[][]
                {
                        {0, 0},
                        {1, 0},
                        {0, 1},
                        {1, 1}
                };

        double[][] target = new double[][]
                {
                        {0},
                        {1},
                        {1},
                        {0},
                };

        neuralNetwork.setTrainingData(input, target);

        neuralNetwork.setDropout(0.0);

        neuralNetwork.setNumberOfProcessThreads(1);

        for (int i = 0; i < 10; i++) {
            neuralNetwork.train(10000);
        }
    }
}
