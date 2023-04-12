package net.porzel.functions;

import java.util.Random;

public abstract class WeightInitialization {
    public abstract void function(double[][][] field);

    public abstract void function(double[][] field);

    /**
     * The weight initialization function HE, which stands for "He-et-al.",
     * is designed to work well with the Rectified Linear Unit (ReLU) activation function.
     * ReLU is a popular choice for deep learning neural networks due to its simplicity and
     * effectiveness in preventing the vanishing gradient problem. HE initialization helps to ensure
     * that the ReLU activation function is operating within its linear region,
     * allowing for efficient and stable learning.
     * */
    public static WeightInitialization HE() {
        return new WeightInitialization() {
            @Override
            public void function(double[][][] field) {
                Random random = new Random();

                for (int layer = 0; layer < field.length; layer++) {
                    double std = Math.sqrt(2.0 / field[layer].length);

                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        for (int weight = 0; weight < field[layer][neuron].length; weight++) {
                            field[layer][neuron][weight] = random.nextGaussian() * std;
                        }
                    }
                }
            }

            @Override
            public void function(double[][] field) {
                Random random = new Random();

                for (int layer = 0; layer < field.length; layer++) {
                    double std = Math.sqrt(2.0 / field[layer].length);

                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        field[layer][neuron] = random.nextGaussian() * std;
                    }
                }
            }
        };
    }

    /**
     * The weight initialization function Lecun, also known as LeCun initialization,
     * is designed to work well with the hyperbolic tangent (tanh) activation function.
     * Tanh is a popular choice for neural networks due to its nonlinearity and its ability
     * to map inputs to outputs between -1 and 1. Lecun initialization helps to ensure that
     * the tanh activation function is operating within its linear region,
     * allowing for efficient and stable learning.
     */
    public static WeightInitialization LECUN() {
        return new WeightInitialization() {
            @Override
            public void function(double[][][] field) {
                Random random = new Random();

                for (int layer = 0; layer < field.length; layer++) {
                    double std = 1.0 / Math.sqrt(field[layer].length);

                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        for (int weight = 0; weight < field[layer][neuron].length; weight++) {
                            field[layer][neuron][weight] = random.nextGaussian() * std;
                        }
                    }
                }
            }

            @Override
            public void function(double[][] field) {
                Random random = new Random();

                for (int layer = 0; layer < field.length; layer++) {
                    double std = 1.0 / Math.sqrt(field[layer].length);

                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        field[layer][neuron] = random.nextGaussian() * std;
                    }
                }
            }
        };
    }

    /**
     *The weight initialization function Glorot, also known as Xavier initialization,
     * is designed to work well with activation functions that have a symmetric S-shape,
     * such as the hyperbolic tangent (tanh) and logistic sigmoid functions.
     * Glorot initialization helps to ensure that the activation function is operating within its linear region,
     * allowing for efficient and stable learning. The idea behind Glorot initialization is to set the scale
     * of the weights based on the number of input and output connections of each neuron, which can help to
     * prevent the vanishing or exploding gradient problems. Glorot initialization is a popular choice for
     * many deep learning applications, and can help to improve the performance and convergence of neural networks.
     */
    public static WeightInitialization GLOROT() {
        return new WeightInitialization() {
            @Override
            public void function(double[][][] field) {
                Random random = new Random();

                for (int layer = 0; layer < field.length; layer++) {
                    double range = Math.sqrt(2.0 / (field[layer].length + 1));

                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        for (int weight = 0; weight < field[layer][neuron].length; weight++) {
                            field[layer][neuron][weight] = random.nextDouble() * 2 * range - range;
                        }
                    }
                }
            }

            @Override
            public void function(double[][] field) {
                Random random = new Random();

                for (int layer = 0; layer < field.length; layer++) {
                    double range = Math.sqrt(2.0 / (field[layer].length + 1));

                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        field[layer][neuron] = random.nextDouble() * 2 * range - range;
                    }
                }
            }
        };
    }

    /**
     * The weight initialization function Xavier, also known as Glorot initialization,
     * is designed to work well with symmetric activation functions such as the hyperbolic
     * tangent (tanh) and logistic sigmoid functions. These activation functions have a symmetric
     * S-shape, with output values ranging from -1 to 1 for tanh and 0 to 1 for sigmoid.
     * Xavier initialization helps to ensure that the activation function is operating
     * within its linear region, allowing for efficient and stable learning. The main idea
     * behind Xavier initialization is to set the scale of the weights based on the number of
     * input and output connections of each neuron, which can help to prevent the vanishing or
     * exploding gradient problems.
     */
    public static WeightInitialization XAVIER() {
        return new WeightInitialization() {
            @Override
            public void function(double[][][] field) {
                Random random = new Random();
                double range = Math.sqrt(6.0 / (field[0].length + field[field.length - 1].length));

                for (int layer = 0; layer < field.length; layer++) {
                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        for (int weight = 0; weight < field[layer][neuron].length; weight++) {
                            field[layer][neuron][weight] = random.nextDouble() * 2 * range - range;
                        }
                    }
                }
            }

            @Override
            public void function(double[][] field) {
                Random random = new Random();
                double range = Math.sqrt(1.0 / (field[0].length + field[field.length - 1].length));

                for (int layer = 0; layer < field.length; layer++) {
                    for (int neuron = 0; neuron < field[layer].length; neuron++) {
                        field[layer][neuron] = random.nextDouble() * 2 * range - range;
                    }
                }
            }
        };
    }
}
