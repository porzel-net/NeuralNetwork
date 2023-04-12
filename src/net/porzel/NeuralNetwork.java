package net.porzel;

import net.porzel.functions.ActivationFunction;
import net.porzel.functions.WeightInitialization;

import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {
    private final double[][][] weights;
    private final double[][] biases;
    private double learningRate;
    private ActivationFunction activationFunction;

    private double[][] trainingDataInput, trainingDataTarget;
    private double[][] testDataInput, testDataTarget;

    private int trainingEpochs = 0;

    private Thread neuralNetworkStatusPrinter;

    public NeuralNetwork(int[] layers) {
        if(layers.length < 3)
            throw new RuntimeException("The network must have at least 3 layers!");

        weights = new double[layers.length - 1][][];
        biases = new double[layers.length - 1][];

        for (int layer = 0; layer < layers.length - 1; layer++) {
            if(layers[layer] < 1)
                throw new RuntimeException("There cant be less than one Neuron in one Layer!");

            weights[layer] = new double[layers[layer + 1]][layers[layer]];
            biases[layer] = new double[layers[layer + 1]];
        }

        WeightInitialization weightInitialization = WeightInitialization.XAVIER();

        weightInitialization.function(weights);
        weightInitialization.function(biases);
    }

    public NeuralNetwork setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    public NeuralNetwork setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuralNetwork setWeightInitializationFunction(WeightInitialization weightInitialization) {
        weightInitialization.function(weights);
        weightInitialization.function(biases);

        return this;
    }

    public double[] propagation(double[] input) {
        if(input.length != weights[0][0].length)
            throw new RuntimeException("The given input length doesnt match with the number of input-neurons!");

        for (int layer = 0; layer < weights.length; layer++) {
            double[] temp = new double[weights[layer].length];
            Arrays.fill(temp, 0);

            for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                    temp[neuron] += input[weight] * weights[layer][neuron][weight];
                }
                temp[neuron] += biases[layer][neuron];
            }

            activationFunction.function(temp);

            input = temp.clone();
        }

        return input;
    }

    public void backPropagation(double[] input, double[] targetOutput) {
        if(input.length != weights[0][0].length)
            throw new RuntimeException("The given input length doesnt match with the number of input-neurons!");

        if(targetOutput.length != weights[weights.length - 1].length)
            throw new RuntimeException("The given target output length doesnt match with the number of output-neurons!");

        double[][] outputs = new double[weights.length][];

        for (int layer = 0; layer < outputs.length; layer++)
            outputs[layer] = new double[weights[layer].length];

        double[] tempInput = input.clone();

        //PROPAGATION
        for (int layer = 0; layer < weights.length; layer++) {
            double[] tempOutput = new double[weights[layer].length];
            Arrays.fill(tempOutput, 0);

            for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                    tempOutput[neuron] += tempInput[weight] * weights[layer][neuron][weight];
                }
                tempOutput[neuron] += biases[layer][neuron];
            }

            activationFunction.function(tempOutput);

            tempInput = tempOutput.clone();
            outputs[layer] = tempOutput.clone();
        }

        //BACKPROPAGATION
        double[][] neuronError = new double[weights.length][];

        for (int layer = 0; layer < neuronError.length; layer++)
            neuronError[layer] = new double[weights[layer].length];

        //ERROR CALCULATING OUTPUT LAYER
        for (int neuron = 0; neuron < neuronError[neuronError.length - 1].length; neuron++) {
            //ERROR CALCULATING OUTPUT LAYER
            neuronError[neuronError.length - 1][neuron] = (targetOutput[neuron] - outputs[outputs.length - 1][neuron]);

            //UPDATE WEIGHTS OUTPUT LAYER
            for (int weight = 0; weight < weights[neuronError.length - 1][neuron].length; weight++) {
                weights[neuronError.length - 1][neuron][weight] += learningRate * neuronError[neuronError.length - 1][neuron] * outputs[neuronError.length - 1 - 1][weight] * activationFunction.derivative(outputs[outputs.length - 1][neuron]);
            }
            biases[neuronError.length - 1][neuron] += neuronError[neuronError.length - 1][neuron] * learningRate * activationFunction.derivative(outputs[outputs.length - 1][neuron]);
        }


        for (int layer = neuronError.length - 2; layer >= 0; layer--) {
            for (int neuron = 0; neuron < neuronError[layer].length; neuron++) {
                //ERROR CALCULATING HIDDEN LAYER
                for (int neuronNextLayer = 0; neuronNextLayer < weights[layer + 1].length; neuronNextLayer++) {
                    neuronError[layer][neuron] += neuronError[layer + 1][neuronNextLayer] * weights[layer + 1][neuronNextLayer][neuron];
                }

                //UPDATE WEIGHTS HIDDEN LAYER
                for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                    if(layer - 1 >= 0)
                        weights[layer][neuron][weight] += learningRate * neuronError[layer][neuron] * outputs[layer - 1][weight] * activationFunction.derivative(outputs[layer][neuron]);
                    else
                        weights[layer][neuron][weight] += learningRate * neuronError[layer][neuron] * input[weight] * activationFunction.derivative(outputs[layer][neuron]);
                }
                biases[layer][neuron] += neuronError[layer][neuron] * learningRate * activationFunction.derivative(outputs[layer][neuron]);
            }
        }

        trainingEpochs += 1;
    }


    //TRAINING METHODS
    public void setTrainingData(double[][] input, double[][] targetOutput) {
        if(input.length != targetOutput.length)
            throw new RuntimeException("The given dataset length doesnt match with the dataset target values!");

        trainingDataInput = input;
        trainingDataTarget = targetOutput;
    }

    public void train(long time) {
        long startTime = System.currentTimeMillis();

        neuralNetworkStatusPrinter = new Thread(new Runnable() {
            int progressBarLength = 60;

            private void updateProgressBar() {
                String output = "";

                output += "\r";

                double percentage = (double) (System.currentTimeMillis() - startTime) / time;

                output += "Neuronal Network   " + Math.round(percentage * 100) + "% [";

                for (int i = 0; i < Math.round(progressBarLength * percentage); i++) {
                    output += "=";
                }

                for (int i = 0; i < Math.round(progressBarLength * (1 - percentage)); i++) {
                    output += " ";
                }

                output += "] " + trainingEpochs + " (" + (System.currentTimeMillis() - startTime) / 1000 + "s / " + time / 1000 + "s)";

                System.out.print(output);

                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                output = "\r";

                for (int i = 0; i < 100; i++) {
                    output += "";
                }

                System.out.print(output);
            }

            @Override
            public void run() {
                while ((System.currentTimeMillis() - startTime < time)) {
                    updateProgressBar();
                };

                updateProgressBar();
                System.out.println("\nCompleted with an accuracy of " + Math.round(accuracy(trainingDataInput, trainingDataTarget) * 100) + "% after " + (int)(time / 1000) + "s and " + trainingEpochs + " epochs");
            }
        });

        neuralNetworkStatusPrinter.start();

        while (System.currentTimeMillis() - startTime < time) {
            train();
        }
    }

    public void train(int epochs) {
        long startTime = System.currentTimeMillis();

        neuralNetworkStatusPrinter = new Thread(new Runnable() {
            int progressBarLength = 60;

            private void updateProgressBar() {
                System.out.print("\r");

                double percentage = (double) trainingEpochs / epochs;

                System.out.print("Neuronal Network   " + Math.round(percentage * 100) + "% [");

                for (int i = 0; i < Math.round(progressBarLength * percentage) - 1; i++) {
                    System.out.print("=");
                }

                System.out.print(">");

                for (int i = 0; i < Math.round(progressBarLength * (1 - percentage)); i++) {
                    System.out.print(" ");
                }

                System.out.print("] " + trainingEpochs + " / " + epochs + " (" + (System.currentTimeMillis() - startTime) / 1000 + "s)");

                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                System.out.print("\r");

                for (int i = 0; i < 100; i++) {
                    System.out.print("");
                }
            }

            @Override
            public void run() {
                while (trainingEpochs < epochs) {
                    updateProgressBar();
                };

                updateProgressBar();
                System.out.println("\nCompleted with an accuracy of " + Math.round(accuracy(trainingDataInput, trainingDataTarget) * 100) + "% after " + (int)((System.currentTimeMillis() - startTime) / 1000) + "s and " + trainingEpochs + " epochs.");
            }
        });

        neuralNetworkStatusPrinter.start();

        trainingEpochs = 0;

        while (trainingEpochs < epochs) {
            train();
        }
    }

    private void train() {
        if(trainingDataTarget == null || trainingDataInput == null)
            throw new RuntimeException("No training data given!");

        int randomDataset = new Random().nextInt(trainingDataInput.length);

        backPropagation(trainingDataInput[randomDataset], trainingDataTarget[randomDataset]);
    }

    public void setTestData(double[][] input, double[][] targetOutput) {
        if(input.length != targetOutput.length)
            throw new RuntimeException("The given dataset length doesnt match with the dataset target values!");

        trainingDataInput = input;
        trainingDataTarget = targetOutput;
    }

    private double accuracy(double[][] input, double[][] target) {
        double absoluteError = 0;

        for (int i = 0; i < input.length; i++) {
            double[] output = propagation(input[i]);

            double error = 0;

            for (int j = 0; j < output.length; j++) {
                error += Math.abs(output[j] - target[i][j]);
            }

            absoluteError += error / output.length;
        }

        return 1 - (absoluteError / input.length);
    }
}