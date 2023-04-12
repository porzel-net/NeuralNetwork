package net.porzel.functions;

public abstract class ActivationFunction {

    public abstract void function(double[] x);

    public abstract double derivative(double x);

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
        };
    }

    /**
     * The TANH activation function is commonly used in neural networks due to its ability to
     * produce outputs between -1 and 1, making it suitable for modeling data that has negative
     * and positive values. Its symmetric shape also allows for better representation of data
     * with both positive and negative features. However, the TANH function suffers from the
     * vanishing gradient problem, where gradients become extremely small, making it difficult
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
        };
    }

}
