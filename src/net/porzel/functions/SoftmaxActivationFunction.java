package net.porzel.functions;

/**

 The SoftmaxActivationFunction class provides methods to calculate the softmax activation function
 and its derivative for a given input array.
 */
public class SoftmaxActivationFunction {

    /**
     This method calculates the softmax function for a given input array x.
     It first exponentiates each element of the array, then normalizes the array
     by dividing each element by the sum of all exponentiated elements.
     @param x the input array
     @return the softmax of the input array
     */
    public static double[] function(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i]);
            sum += x[i];
        }
        for (int i = 0; i < x.length; i++)
            x[i] /= sum;

        return x;
    }

    /**
     This method calculates the derivative of the softmax function for a given input array x.
     The derivative is calculated element-wise using the formula:
     softmax'(z_i) = softmax(z_i) * (1 - softmax(z_i)), if i = j
     softmax'(z_i) = -softmax(z_i) * softmax(z_j), if i != j
     @param x the input array
     @return the derivative of the softmax function at the input array
     */
    public static double[] softmaxDerivative(double[] x) {
        double[] y = function(x);
        double[] derivatives = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double sum = 0;
            for (int j = 0; j < x.length; j++) {
                if (i == j) {
                    derivatives[i] += y[i] * (1 - y[i]);
                } else {
                    derivatives[i] += -y[i] * y[j];
                }
            }
        }
        return derivatives;
    }
}
