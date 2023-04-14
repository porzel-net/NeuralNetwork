package net.porzel.functions;

public abstract class LossFunction {
    public abstract double function(double[] output, double[] targetOutput);

    /**

     Resolves the loss function given a string representation of the function name.
     @param function the string representation of the loss function name
     @return the corresponding loss function object
     @throws RuntimeException if the loss function name cannot be resolved
     */
    public static LossFunction resolveLossFunction(String function) {
        return switch (function) {
            case "MEAN_SQUARED_ERROR" -> LossFunction.MEAN_SQUARED_ERROR();
            case "MEAN_ABSOLUTE_ERROR" -> LossFunction.MEAN_ABSOLUTE_ERROR();
            case "CROSS_ENTROPY_LOSS" -> LossFunction.CROSS_ENTROPY_LOSS();
            case "BINARY_CROSS_ENTROPY_LOSS" -> LossFunction.BINARY_CROSS_ENTROPY_LOSS();
            case "HINGE_LOSS" -> LossFunction.HINGE_LOSS();
            case "LOG_LIKELIHOOD_LOSS" -> LossFunction.LOG_LIKELIHOOD_LOSS();
            case "NONE" -> null;
            default -> throw new RuntimeException("Loss function couldn't be resolved!");
        };
    }

    /**
     This Java method returns an instance of a LossFunction interface implementing the Mean Squared Error (MSE) loss function.
     The MSE loss function is commonly used in regression tasks to measure the error between the predicted output and the actual target output.
     The loss function calculates the squared difference between the predicted and target output for each training example, and then calculates the mean of these values over the entire training set. The higher the value of the MSE, the worse the performance of the neural network model. The MSE is used as a cost function to optimize the parameters of a neural network during the training process using backpropagation.
     @return an instance of a LossFunction interface implementing the Mean Squared Error (MSE) loss function.
     */
    public static LossFunction MEAN_SQUARED_ERROR() {
        return new LossFunction() {
            @Override
            public double function(double[] output, double[] targetOutput) {
                double error = 0;

                for (int i = 0; i < output.length; i++)
                    error += Math.pow(targetOutput[i] - output[i], 2);

                return error / output.length;
            }

            @Override
            public String toString() {
                return "MEAN_SQUARED_ERROR";
            }
        };
    }

    /**
     A loss function that computes the mean absolute error between the output and target output of a neural network.
     The mean absolute error is defined as the average of the absolute differences between the output and target output values.
     This loss function is used to evaluate the performance of a neural network in regression tasks.
     */
    public static LossFunction MEAN_ABSOLUTE_ERROR() {
        return new LossFunction() {
            @Override
            public double function(double[] output, double[] targetOutput) {
                double loss = 0.0;

                for (int i = 0; i < targetOutput.length; i++) {
                    loss += Math.abs(output[i] - targetOutput[i]);
                }

                return loss / targetOutput.length;
            }

            @Override
            public String toString() {
                return "MEAN_ABSOLUTE_ERROR";
            }
        };
    }

    /**
     This Java method returns an instance of a LossFunction interface implementing the Cross-Entropy Loss function, which is commonly used in machine learning tasks related to binary classification.
     The Cross-Entropy Loss function measures the error between the predicted output and the actual target output, where the target output is represented by a binary vector of 0s and 1s. The loss function penalizes large errors in prediction by assigning higher loss values to incorrect predictions. This helps to train a neural network model to minimize the error and improve the accuracy of the predictions.
     @return an instance of a LossFunction interface implementing the Cross-Entropy Loss function.
     */
    public static LossFunction CROSS_ENTROPY_LOSS() {
        return new LossFunction() {
            @Override
            public double function(double[] output, double[] targetOutput) {
                double loss = 0.0;

                for (int i = 0; i < targetOutput.length; i++) {
                    loss += targetOutput[i] * Math.log(output[i]);
                }

                return -loss / targetOutput.length;
            }

            @Override
            public String toString() {
                return "CROSS_ENTROPY_LOSS";
            }
        };
    }

    /**
     This class implements the binary cross-entropy loss function, which is commonly used in neural networks for
     binary classification tasks.
     */
    public static LossFunction BINARY_CROSS_ENTROPY_LOSS() {
        return new LossFunction() {
            @Override
            public double function(double[] output, double[] targetOutput) {
                double loss = 0.0;

                for (int i = 0; i < targetOutput.length; i++) {
                    loss += targetOutput[i] * Math.log(output[i]) + (1 - targetOutput[i]) * Math.log(1 - output[i]);
                }

                return -loss / targetOutput.length;
            }

            @Override
            public String toString() {
                return "BINARY_CROSS_ENTROPY_LOSS";
            }
        };
    }

    /**
     This Java comment is defining the HINGE_LOSS function for a neural network as a static method that returns a
     LossFunction object. The HINGE_LOSS function is a loss function that is commonly used in binary classification
     tasks in machine learning. It is used to measure the difference between the predicted output of a model and the
     true output, and is particularly effective for models that output binary values (i.e., 0 or 1).
     */
    public static LossFunction HINGE_LOSS() {
        return new LossFunction() {
            @Override
            public double function(double[] output, double[] targetOutput) {
                double loss = 0.0;

                for (int i = 0; i < targetOutput.length; i++) {
                    loss += Math.max(0, 1 - targetOutput[i] * output[i]);
                }

                return loss / targetOutput.length;
            }

            @Override
            public String toString() {
                return "HINGE_LOSS";
            }
        };
    }

    /**
     This Java comment is for the implementation of the LOG_LIKELIHOOD_LOSS() function, which is a loss function commonly used in neural networks.
     The LOG_LIKELIHOOD_LOSS function is used in classification tasks where the output is a probability distribution over classes, and the target output is a one-hot encoding of the true class.
     */
    public static LossFunction LOG_LIKELIHOOD_LOSS() {
        return new LossFunction() {
            @Override
            public double function(double[] output, double[] targetOutput) {
                double loss = 0.0;
                double sumExp = 0.0;

                for (int i = 0; i < output.length; i++) {
                    sumExp += Math.exp(output[i]);
                }

                for (int i = 0; i < targetOutput.length; i++) {
                    if (targetOutput[i] == 1) {
                        loss -= output[i] - Math.log(sumExp);
                    }
                }

                return loss;
            }

            @Override
            public String toString() {
                return "LOG_LIKELIHOOD_LOSS";
            }
        };
    }
}
