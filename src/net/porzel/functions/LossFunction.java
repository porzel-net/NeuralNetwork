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
            case "CROSS_ENTROPY_LOSS" -> LossFunction.CROSS_ENTROPY_LOSS();
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

    public static double binaryCrossEntropyLoss(double[] output, double[] target) {
        int n = output.length;
        double loss = 0;
        for (int i = 0; i < n; i++) {
            loss += target[i] * Math.log(output[i]) + (1 - target[i]) * Math.log(1 - output[i]);
        }
        loss = -loss / n;
        return loss;
    }

    /*public static LossFunction HINGE_LOSS() {
        return new LossFunction() {
            @Override
            public double function(double output, double targetOutput) {
                return Math.max(0, 1 - targetOutput * output);
            }

            @Override
            public String toString() {
                return "HINGE_LOSS";
            }
        };
    }

    public static LossFunction LOG_LIKELIHOOD_LOSS() {
        return new LossFunction() {
            @Override
            public double function(double output, double targetOutput) {
                return -targetOutput * Math.log(output);
            }

            @Override
            public String toString() {
                return "LOG_LIKELIHOOD_LOSS";
            }
        };
    }


    public static LossFunction DIFFERENTIAL_ERROR() {
        return new LossFunction() {
            @Override
            public double function(double output, double targetOutput) {
                return targetOutput - output;
            }

            @Override
            public String toString() {
                return "DIFFERENTIAL_ERROR";
            }
        };
    }*/
}
