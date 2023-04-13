package net.porzel.functions;

import java.util.Arrays;

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
     */
    public static void function(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.exp(x[i]);
            sum += x[i];
        }
        for (int i = 0; i < x.length; i++)
            x[i] /= sum;
    }

    /**

     This method calculates the derivative of a given function at the given point.
     It takes in an array of doubles as input, which represents the point at which
     the derivative is to be evaluated. The function evaluated at the point is stored
     in a separate double array.
     The derivative is computed using the formula for the derivative of a sigmoid
     function. The derivative for each input feature is computed using a nested for loop.
     For each input feature, the derivative is computed as the product of the output value
     of the function at that point, and 1 minus the output value of the function at the same point.
     The computed derivatives are stored in a new double array and returned as output.
     @param x an array of doubles representing the point at which the derivative is to be evaluated
     */
    public static void derivative(double[] x) {
        double[] y = x.clone();
        function(y);

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                if (i == j) {
                    x[i] = y[i] * (1 - y[j]);
                }
            }
        }
    }
}
