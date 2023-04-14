package net.porzel.functions;

/**
 This is an abstract class for an activation function used in neural networks.
 The activation function is responsible for introducing non-linearity to the network
 and is used to determine the output of a node given its input.
 */
public abstract class ActivationFunction {
    /**
     This method represents the activation function that maps the input to the output of a neuron.
     The input to the function is an array of doubles and the function should modify the values in
     the array to represent the output of the activation function. The output of the function
     should be in the range [0,1] for sigmoidal functions or [-1,1] for hyperbolic tangent
     functions, depending on the implementation.
     @param x the input to the activation function
     */
    public abstract void function(double[] x);

    /**
     This method returns the derivative of the activation function with respect to its input x.
     The derivative is used in backpropagation to calculate the error signal propagated
     from the output layer to the input layer. The input to the function is a double and
     the output should be in the range [0,1] for sigmoidal functions or [-1,1] for
     hyperbolic tangent functions, depending on the implementation.
     @param x the input to the activation function
     @return the derivative of the activation function at x
     */
    public abstract double derivative(double x);

    /**

     Resolves the activation function based on the given string input.
     @param function The name of the activation function to be resolved. The valid inputs are "RELU", "LEAKY_RELU", "SIGMOID", and "TANH".
     @return The corresponding ActivationFunction enum value.
     @throws RuntimeException If the input string is not one of the valid activation functions.
     */
    public static ActivationFunction resolveActivationFunction(String function) {
        if (function.startsWith("RELU")) return ActivationFunction.RELU();
        else if (function.startsWith("LEAKY_RELU")) return ActivationFunction.LEAKY_RELU();
        else if (function.startsWith("ELU"))
            try { return ActivationFunction.ELU(Double.parseDouble(function.substring(function.indexOf("(") + 1, function.indexOf(")")))); }
            catch (NumberFormatException e) { throw new RuntimeException("Could not parse ELU parameter as a double!"); }
        else if (function.startsWith("SIGMOID")) return ActivationFunction.SIGMOID();
        else if (function.startsWith("TANH")) return ActivationFunction.TANH();
        else throw new RuntimeException("Activation function couldn't be resolved!");
    }

    /**
     * ReLU is a popular activation function in deep learning. Its advantages include computational
     * efficiency, speed, and sparsity, which can help with model interpretability and reduce
     * overfitting. However, it has some issues like dead neurons and dying ReLUs, which can lead
     * to a loss of model capacity and make it unsuitable for certain types of data.
    * */
    public static ActivationFunction RELU(){
        return new ActivationFunction() {
            @Override
            public void function(double[] x) {
                for (int i = 0; i < x.length; i++)
                    x[i] = Math.max(0, x[i]);
            }

            @Override
            public double derivative(double x) {
                return x > 0 ? 1 : 0;
            }

            @Override
            public String toString() {
                return "RELU()";
            }
        };
    }

    /**
    * The Leaky ReLU activation function is a variant of the popular Rectified Linear Unit (ReLU)
     * activation function. It is designed to address the "dying ReLU" problem that occurs when the
     * ReLU function outputs zero for negative input values, effectively killing the activation of
     * that neuron. The Leaky ReLU function assigns a small slope to negative input values, which
     * allows for non-zero gradients and thus avoids the "dying ReLU" problem. The main advantage of
     * the Leaky ReLU function is that it can help prevent the vanishing gradient problem, which
     * can hinder the training of deep neural networks. However, the main disadvantage of the Leaky
     * ReLU function is that it introduces an additional hyperparameter (the slope of the negative part),
     * which can be difficult to tune and may require additional computational resources to optimize.
     */
    public static ActivationFunction LEAKY_RELU() {
        return new ActivationFunction() {
            @Override
            public void function(double[] x) {
                for (int i = 0; i < x.length; i++)
                    x[i] = x[i] > 0 ? x[i] : 0.01 * x[i];
            }

            @Override
            public double derivative(double x) {
                return x > 0 ? 1 : 0.01;
            }

            @Override
            public String toString() {
                return "LEAKY_RELU()";
            }
        };
    }

    /**
     * The sigmoid activation function is a commonly used non-linear activation function
     * in neural networks. One of the advantages of sigmoid is that it produces values
     * between 0 and 1, which can be useful in certain cases where we need to model
     * probabilities. Additionally, it has a smooth gradient which allows for more stable
     * training. However, the main disadvantage of sigmoid is that it suffers from the
     * vanishing gradient problem, which can make it difficult to train deep neural networks.
     * This is because the gradient becomes very small as the input approaches the extremes
     * of the function, leading to slow learning and potential convergence issues. As a result,
     * other activation functions like ReLU and its variants have become more popular in recent years.
     * */
    public static ActivationFunction SIGMOID(){
        return new ActivationFunction() {

            @Override
            public void function(double[] x) {
                for (int i = 0; i < x.length; i++)
                    x[i] = 1 / (1 + Math.exp(-x[i]));
            }

            @Override
            public double derivative(double x) {
                return x * (1 - x);
            }

            @Override
            public String toString() {
                return "SIGMOID()";
            }
        };
    }

    /**
     * The TANH activation function is commonly used in neural networks due to its ability to
     * produce outputs between -1 and 1, making it suitable for modeling data that has negative
     * and positive values. Its symmetric shape also allows for better representation of data
     * with both positive and negative features. However, the TANH function suffers from the
     * vanishing gradient problem, where gradients become extreme tiny, making it difficult
     * for the network to learn. Additionally, TANH tends to saturate when inputs are large,
     * which can cause outputs to remain at extreme values, leading to slower learning and
     * potential numerical instability.
     * */
    public static ActivationFunction TANH(){
        return new ActivationFunction() {

            @Override
            public void function(double[] x) {
                for (int i = 0; i < x.length; i++)
                    x[i] = Math.tanh(x[i]);
            }

            @Override
            public double derivative(double x) {
                return 1 - Math.pow(Math.tanh(x), 2);
            }

            @Override
            public String toString() {
                return "TANH()";
            }
        };
    }

    /**

     The Exponential Linear Unit (ELU) activation function is commonly used in neural networks due to
     its ability to alleviate the vanishing gradient problem present in other activation functions,
     such as the hyperbolic tangent (TANH). ELU produces outputs between -1 and infinity, and its
     smooth, continuous shape allows for better representation of data with both positive and negative
     features. However, ELU is computationally more expensive than other activation functions, and
     its parameter alpha can be difficult to tune. Despite these drawbacks, ELU is a popular choice
     for neural networks due to its effectiveness in reducing the vanishing gradient problem and
     improving learning speed.
     */
    public static ActivationFunction ELU(double alpha){
        return new ActivationFunction() {
            @Override
            public void function(double[] x) {
                for (int i = 0; i < x.length; i++) {
                    if (x[i] < 0) {
                        x[i] = alpha * (Math.exp(x[i]) - 1);
                    }
                }
            }

            @Override
            public double derivative(double x) {
                if (x >= 0) {
                    return 1;
                } else {
                    return alpha * Math.exp(x);
                }
            }

            @Override
            public String toString() {
                return "ELU(" + alpha + ")";
            }
        };
    }
}
