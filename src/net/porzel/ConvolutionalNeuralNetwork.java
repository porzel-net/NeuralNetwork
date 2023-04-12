package net.porzel;

import net.porzel.functions.ActivationFunction;
import net.porzel.functions.CostFunction;
import net.porzel.functions.LossFunction;
import net.porzel.functions.WeightInitialization;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class ConvolutionalNeuralNetwork {
    //DATA
    private final double[][][] weights;
    private final double[][] biases;
    private double learningRate = 0.01;
    private double[] loses;

    //FUNCTIONS
    private ActivationFunction activationFunction;
    private LossFunction lossFunction;
    private CostFunction costFunction;

    private double dropout = 0;

    //TRAINING-, TEST- DATA
    private double[][] trainingDataInput, trainingDataTarget;
    private double[][] testDataInput, testDataTarget;

    //SETTINGS
    int numberOfThreads = 1;

    //INFO
    private int totalTrainedEpochs = 0;
    private int trainedEpochs = 0;
    private Thread neuralNetworkStatusPrinter;
    private final int[] layers;

    //THREADS
    public class BackPropagationNeuronWeightAdaptionRunnable extends Thread implements Runnable {
        final int layer, from, to;
        final double[][] neuronError, outputs;
        final double[] input;

        public BackPropagationNeuronWeightAdaptionRunnable(int layer, int from, int to, double[][] neuronError, double[][] outputs, double[] input) {
            this.layer = layer;
            this.from = from;
            this.to = to;
            this.neuronError = neuronError;
            this.outputs = outputs;
            this.input = input;
        }


        @Override
        public void run() {
            for (int neuron = from; neuron < to; neuron++) {
                //ERROR CALCULATING HIDDEN LAYER
                for (int neuronNextLayer = 0; neuronNextLayer < weights[layer + 1].length; neuronNextLayer++) {
                    neuronError[layer][neuron] += neuronError[layer + 1][neuronNextLayer] * weights[layer + 1][neuronNextLayer][neuron];
                }

                //UPDATE WEIGHTS HIDDEN LAYER
                for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                    if (layer - 1 >= 0)
                        weights[layer][neuron][weight] += learningRate * neuronError[layer][neuron] * outputs[layer - 1][weight] * activationFunction.derivative(outputs[layer][neuron]);
                    else
                        weights[layer][neuron][weight] += learningRate * neuronError[layer][neuron] * input[weight] * activationFunction.derivative(outputs[layer][neuron]);
                }
                biases[layer][neuron] += neuronError[layer][neuron] * learningRate * activationFunction.derivative(outputs[layer][neuron]);
            }
        }
    }


    /**
     Constructs a neural network with the specified layer sizes.
     @param layers an array of integers specifying the number of neurons in each layer. Must have at least 3 layers.
     @throws RuntimeException if the number of layers is less than 3 or if any layer has fewer than 1 neuron
     */
    public ConvolutionalNeuralNetwork(int[] layers) {
        if (layers.length < 3)
            throw new RuntimeException("The network must have at least 3 layers!");
        this.layers = layers;

        weights = new double[layers.length - 1][][];
        biases = new double[layers.length - 1][];

        for (int layer = 0; layer < layers.length - 1; layer++) {
            if (layers[layer] < 1)
                throw new RuntimeException("There cant be less than one Neuron in one Layer!");

            weights[layer] = new double[layers[layer + 1]][layers[layer]];
            biases[layer] = new double[layers[layer + 1]];
        }

        WeightInitialization weightInitialization = WeightInitialization.XAVIER();

        weightInitialization.function(weights);
        weightInitialization.function(biases);

        activationFunction = ActivationFunction.LEAKY_RELU();
    }


    private ConvolutionalNeuralNetwork(int[] layers, double[][][] weights, double[][] biases, double learningRate, double dropout, ActivationFunction activationFunction, LossFunction lossFunction, int totalTrainedEpochs) {
        this.layers = layers;
        this.weights = weights;
        this.biases = biases;
        this.learningRate = learningRate;
        this.dropout = dropout;
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
        this.totalTrainedEpochs = totalTrainedEpochs;
    }

    /**
     Sets the activation function for the neural network.
     @param activationFunction the activation function to be used
     @return this convolutional neural network object for method chaining
     */
    public ConvolutionalNeuralNetwork setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    /**
     This method sets the loss function to be used by the neural network. The loss function is an important part of training a neural network, as it measures how well the network is performing. The loss function is typically a mathematical function that compares the predicted output of the network with the actual output, and returns a value that indicates the degree of error.
     @param lossFunction the loss function to be set
     @return the current convolutional neural network object, for method chaining
     */
    public ConvolutionalNeuralNetwork setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;

        if(costFunction == null)
            costFunction = CostFunction.MEDIAN();

        return this;
    }

    /**
     Sets the cost function used by this convolutional neural network. The cost function is used to compute the overall cost of the network over all of the training data.
     @param costFunction The cost function to use.
     @return This convolutional neural network with the updated cost function.
     */
    public ConvolutionalNeuralNetwork setCostFunction(CostFunction costFunction) {
        this.costFunction = costFunction;

        return this;
    }

    /**
     Sets the learning rate of the neural network.
     @param learningRate the new learning rate to be set
     @return the modified convolutional neural network object with updated learning rate
     */
    public ConvolutionalNeuralNetwork setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    /**
     Sets the weight initialization function to be used by the neural network.
     The specified function is applied to both the weights and biases of the network.
     @param weightInitialization the weight initialization function to be used
     @return the modified convolutional neural network with the updated weight initialization
     */
    public ConvolutionalNeuralNetwork setWeightInitializationFunction(WeightInitialization weightInitialization) {
        weightInitialization.function(weights);
        weightInitialization.function(biases);

        return this;
    }


    /**
     This method takes an array of input values and performs forward propagation
     through a neural network using the stored weights and biases.
     @param input The input values for the neural network.
     @return The output values of the neural network after performing forward propagation.
     @throws RuntimeException if the given input length doesn't match the number of input-neurons in the first layer.
     */
    public double[] propagation(double[] input) {
        // Check if the input length matches the number of input neurons in the first layer.
        if (input.length != weights[0][0].length)
            throw new RuntimeException("The given input length doesnt match with the number of input-neurons!");

        // Perform forward propagation through the neural network.
        for (int layer = 0; layer < weights.length; layer++) {
            // Initialize an array to store the values of the current layer.
            double[] temp = new double[weights[layer].length];
            Arrays.fill(temp, 0);

            // Calculate the values of the neurons in the current layer.
            for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                    temp[neuron] += input[weight] * weights[layer][neuron][weight];
                }
                temp[neuron] += biases[layer][neuron];
            }

            // Apply the activation function to the values of the current layer.
            activationFunction.function(temp);

            // Set the input values for the next layer to be the values of the current layer.
            input = temp.clone();
        }

        return input;
    }


    /**
     Implements the backpropagation algorithm for training a neural network.
     @param input the input values for the neural network
     @param targetOutput the target output values for the neural network
     @throws RuntimeException if the length of the input array does not match the number of input neurons or
     the length of the target output array does not match the number of output neurons
     */
    public void backPropagation(double[] input, double[] targetOutput) {
        if (input.length != weights[0][0].length)
            throw new RuntimeException("The given input length doesnt match with the number of input-neurons!");

        if (targetOutput.length != weights[weights.length - 1].length)
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

            if (dropout > 0 && layer < weights.length - 2) {
                int numNeuronsToDrop = (int) Math.round(dropout * weights[layer].length);
                int[] neuronsToDrop = new Random().ints(0, weights[layer].length).distinct().limit(numNeuronsToDrop).toArray();
                for (int neuron : neuronsToDrop) {
                    tempOutput[neuron] = 0;
                }
                tempOutput = Arrays.stream(tempOutput).map(x -> x / (1 - dropout)).toArray();
            }

            tempInput = tempOutput.clone();
            outputs[layer] = tempOutput.clone();
        }

        //BACKPROPAGATION
        double[][] neuronError = new double[weights.length][];

        for (int layer = 0; layer < neuronError.length; layer++)
            neuronError[layer] = new double[weights[layer].length];

        double networkCost = 1;

        if(lossFunction != null) {
            costFunction.addLoss(lossFunction.function(outputs[outputs.length - 1], targetOutput));
            networkCost = costFunction.getCost();
        }

        //ERROR CALCULATING OUTPUT LAYER
        for (int neuron = 0; neuron < neuronError[neuronError.length - 1].length; neuron++) {
            //ERROR CALCULATING OUTPUT LAYER
            neuronError[neuronError.length - 1][neuron] = (targetOutput[neuron] - outputs[outputs.length - 1][neuron]) * networkCost;

            //UPDATE WEIGHTS OUTPUT LAYER
            for (int weight = 0; weight < weights[neuronError.length - 1][neuron].length; weight++) {
                weights[neuronError.length - 1][neuron][weight] += learningRate * neuronError[neuronError.length - 1][neuron] * outputs[neuronError.length - 1 - 1][weight] * activationFunction.derivative(outputs[outputs.length - 1][neuron]);
            }
            biases[neuronError.length - 1][neuron] += neuronError[neuronError.length - 1][neuron] * learningRate * activationFunction.derivative(outputs[outputs.length - 1][neuron]);
        }


        for (int layer = neuronError.length - 2; layer >= 0; layer--) {
            Thread[] backPropagationThreads = new Thread[numberOfThreads];

            double neuronRange = (double) weights[layer].length / numberOfThreads;

            //INIT THREADS
            for (int thread = 0; thread < numberOfThreads; thread++) {
                backPropagationThreads[thread] = new Thread(new BackPropagationNeuronWeightAdaptionRunnable(layer, (int)Math.round(neuronRange * thread), (int)Math.round(neuronRange * (thread + 1)), neuronError, outputs, input));

                backPropagationThreads[thread].start();
            }

            try {
                //WAIT UNTIL THREADS FINISHED
                for (int thread = 0; thread < numberOfThreads; thread++)
                    backPropagationThreads[thread].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        totalTrainedEpochs += 1;
        trainedEpochs += 1;
    }


    /**
     Trains the neural network for the specified number of epochs.
     @param epochs the number of epochs to train for
     @throws RuntimeException if no training data is provided
     */
    public void train(int epochs) {
        if (trainingDataTarget == null || trainingDataInput == null)
            throw new RuntimeException("No training data given!");

        long startTime = System.currentTimeMillis();

        trainedEpochs = 0;

        neuralNetworkStatusPrinter = new Thread(new Runnable() {
            int progressBarLength = 60;

            private void updateProgressBar() {
                String output = "";

                output += "\r";

                double percentage = (double) trainedEpochs / epochs;

                output += "Neuronal Network   " + Math.round(percentage * 100) + "% [";

                for (int i = 0; i < Math.round(progressBarLength * percentage) - 1; i++) {
                    output += "=";
                }

                output += ">";

                for (int i = 0; i < Math.round(progressBarLength * (1 - percentage)); i++) {
                    output += " ";
                }

                output += "] " + trainedEpochs + " / " + epochs + " (" + (System.currentTimeMillis() - startTime) / 1000 + "s)";

                System.out.print(output);

                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                output = "";

                output += "\r";

                for (int i = 0; i < 100; i++) {
                    output += "";
                }

                System.out.print(output);
            }

            @Override
            public void run() {
                while (trainedEpochs < epochs) {
                    updateProgressBar();
                }

                updateProgressBar();

                double accuracy = (testDataInput == null) ? Math.round(accuracy(trainingDataInput, trainingDataTarget) * 100) : Math.round(accuracy(testDataInput, testDataTarget) * 100);

                System.out.println("\nCompleted with an accuracy of " + accuracy + "% after " + (int) ((System.currentTimeMillis() - startTime) / 1000) + "s and " + totalTrainedEpochs + " epochs.");
            }
        });

        neuralNetworkStatusPrinter.start();

        while (trainedEpochs < epochs) {
            train();
        }

        while (neuralNetworkStatusPrinter.isAlive()) {}
    }

    private void train() {
        if (trainingDataTarget == null || trainingDataInput == null)
            throw new RuntimeException("No training data given!");

        int randomDataset = new Random().nextInt(trainingDataInput.length);

        backPropagation(trainingDataInput[randomDataset], trainingDataTarget[randomDataset]);
    }

    //SETTER & GETTER

    /**
     Sets the dropout percentage for the neural network.
     Dropout is a technique used to prevent overfitting in neural networks by randomly
     dropping out (i.e., setting to zero) neurons during training. This method sets the
     percentage of neurons that should be randomly dropped out during training.
     @param percentage the percentage of neurons to drop out during training (between 0.0 and 1.0)
     */
    public void setDropout(double percentage) {
        dropout = percentage;
    }

    /**
     Sets the number of process threads to be used for parallel processing in the neural network.
     @param numberOfThreads the number of threads to be used
     */
    public void setNumberOfProcessThreads(int numberOfThreads) {
        this.numberOfThreads = numberOfThreads;
    }

    /**
     Sets the training data used for the neural network training.
     The input and targetOutput data must be of the same length.
     @param input the input data for training as a 2D double array
     @param targetOutput the target output data for training as a 2D double array
     @throws RuntimeException if the length of the input and targetOutput data doesn't match
     */
    public void setTrainingData(double[][] input, double[][] targetOutput) {
        if (input.length != targetOutput.length)
            throw new RuntimeException("The given dataset length doesnt match with the dataset target values!");

        trainingDataInput = input;
        trainingDataTarget = targetOutput;
    }

    public void setTestData(double[][] input, double[][] targetOutput) {
        if (input.length != targetOutput.length)
            throw new RuntimeException("The given dataset length doesnt match with the dataset target values!");

        testDataInput = input;
        testDataTarget = targetOutput;
    }

    /**
     Calculates the accuracy of the neural network on the given input and target data.
     Accuracy is defined as 1 minus the mean absolute error of the neural network's predictions compared to the target values.
     @param input the input data to evaluate the accuracy on
     @param target the target data to compare the neural network's predictions against
     @return the accuracy of the neural network as a value between 0 and 1, where 1 is a perfect match and 0 is no match
     */
    public double accuracy(double[][] input, double[][] target) {
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

    /**
     Saves the current state of the neural network to a file.
     The file format consists of the number of layers, the size of each layer,
     the weights and biases for each neuron in each layer, the learning rate,
     the activation function used, the dropout rate, and the total number of epochs
     trained on the network.
     @param file the file to save the network state to
     */
    public void save(File file) {
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(file))) {
            out.writeInt(layers.length);

            for (int layer = 0; layer < layers.length; layer++) {
                out.writeInt(layers[layer]);
            }

            for (int layer = 0; layer < weights.length; layer++) {
                for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                    for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                        out.writeDouble(weights[layer][neuron][weight]);
                    }
                }
            }

            for (int layer = 0; layer < biases.length; layer++) {
                for (int neuron = 0; neuron < biases[layer].length; neuron++) {
                    out.writeDouble(biases[layer][neuron]);
                }
            }

            out.writeDouble(learningRate);
            out.writeUTF(activationFunction.toString());

            if(lossFunction == null)
                out.writeUTF("NONE");
            else
                out.writeUTF(lossFunction.toString());

            out.writeDouble(dropout);
            out.writeInt(totalTrainedEpochs);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     Loads a pre-trained neural network from a file.
     The file should contain the following information in order:
     The number of layers in the neural network (as an integer)
     The number of neurons in each layer (as an integer array)
     The weights for each connection in the neural network (as a 3D double array)
     The biases for each neuron in the neural network (as a 2D double array)
     The learning rate used during training (as a double)
     The activation function used in the neural network (as an ActivationFunction object)
     The dropout rate used during training (as a double)
     The total number of epochs trained (as an integer)
     @param file the file from which to load the neural network
     @return the pre-trained neural network
     @throws RuntimeException if there is an error reading from the file
     */
    public static ConvolutionalNeuralNetwork load(File file)  {
        try (DataInputStream in = new DataInputStream(new FileInputStream(file))) {
            int[] layers = new int[in.readInt()];
            double[][][] weights = new double[layers.length - 1][][];
            double[][] biases = new double[layers.length - 1][];

            for (int layer = 0; layer < layers.length; layer++) {
                layers[layer] = in.readInt();
            }

            for (int layer = 0; layer < layers.length - 1; layer++) {
                weights[layer] = new double[layers[layer + 1]][layers[layer]];
                biases[layer] = new double[layers[layer + 1]];
            }

            for (int layer = 0; layer < weights.length; layer++) {
                for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                    for (int weight = 0; weight < weights[layer][neuron].length; weight++) {
                        weights[layer][neuron][weight] = in.readDouble();
                    }
                }
            }

            for (int layer = 0; layer < biases.length; layer++) {
                for (int neuron = 0; neuron < biases[layer].length; neuron++) {
                    biases[layer][neuron] = in.readDouble();
                }
            }

            double learningRate = in.readDouble();
            ActivationFunction activationFunction = ActivationFunction.resolveActivationFunction(in.readUTF());
            LossFunction lossFunction = LossFunction.resolveLossFunction(in.readUTF());
            double dropout = in.readDouble();
            int totalTrainedEpochs = in.readInt();


            return new ConvolutionalNeuralNetwork(layers, weights, biases, learningRate, dropout, activationFunction, lossFunction, totalTrainedEpochs);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }
}