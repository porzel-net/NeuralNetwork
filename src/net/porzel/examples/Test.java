package net.porzel.examples;

import net.porzel.ConvolutionalNeuralNetwork;
import net.porzel.functions.ActivationFunction;
import net.porzel.functions.CostFunction;
import net.porzel.functions.LossFunction;
import net.porzel.functions.WeightInitialization;

import java.io.File;
import java.util.Arrays;

public class Test {
    public static void main(String[] args) {
        ConvolutionalNeuralNetwork neuralNetwork = new ConvolutionalNeuralNetwork(new int[] {2, 16, 16, 16, 1})
                .setLearningRate(0.015)
                .setActivationFunction(ActivationFunction.LEAKY_RELU())
                .setWeightInitializationFunction(WeightInitialization.HE())
                .setLossFunction(LossFunction.CROSS_ENTROPY_LOSS())
                .setCostFunction(CostFunction.MEDIAN());

        neuralNetwork.setLastLayerSoftmax(true);



        double[][] input = new double[][]
                {
                        {1.2, 3.5},
                        {2.3, 4.5},
                        {1.8, 2.5},
                        {2.5, 3.8},
                        {2.0, 3.0}
                };

        double[][] target = new double[][]
                {
                        {0},
                        {1},
                        {0},
                        {1},
                        {0}
                };

        double[][] testInput = new double[][]
                {
                        {1.0, 3.0},
                        {2.2, 4.2},
                        {1.5, 2.8},
                        {2.8, 3.2},
                        {1.9, 3.1}
                };

        double[][] testTarget = new double[][]
                {
                        {0},
                        {1},
                        {0},
                        {1},
                        {0}
                };

        neuralNetwork.setTrainingData(input, target);
        neuralNetwork.setTestData(testInput, testTarget);

        for (int i = 0; i < 3; i++) {
            neuralNetwork.train(100000);
        }

        System.out.println(neuralNetwork.accuracy(input, target));
    }
}
