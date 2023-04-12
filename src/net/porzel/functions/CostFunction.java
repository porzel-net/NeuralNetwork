package net.porzel.functions;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public abstract class CostFunction {
    public abstract void addLoss(double value);

    public double losses;
    public int numberOfLosses;

    /**
     * Returns the cost of the cost function.
     *
     * @return the cost of the cost function
     */
    public double getCost() {
        return losses / numberOfLosses;
    }

    /**
     * Resolves the loss function given a string representation of the function name.
     *
     * @param function the string representation of the cost function name
     * @return the corresponding cost function object
     * @throws RuntimeException if the cost function name cannot be resolved
     */
    public static CostFunction resolveCostFunction(String function) {
        if (function.startsWith("MEAN_SQUARED_ERROR")) {
            return MEAN_SQUARED_ERROR().init(extractLosses(function), extractNumberOfLosses(function));
        } else if (function.startsWith("MEDIAN")) {
            return MEDIAN().init(extractLosses(function), extractNumberOfLosses(function));
        } else if (function.equals("NONE")) {
            return null;
        } else throw new RuntimeException("Cost function couldn't be resolved!");
    }

    /**
     * Returns a new instance of the Mean Squared Error (MSE) cost function.
     * MSE is commonly used in neural networks for regression problems,
     * where the goal is to predict a continuous numerical output.
     * The MSE cost function calculates the average squared difference between the predicted output
     * and the true output across all training examples in the dataset.
     * @return a new instance of the MSE cost function
     */
    public static CostFunction MEAN_SQUARED_ERROR() {
        return new CostFunction() {
            @Override
            public void addLoss(double value) {
                losses += Math.pow(value, 2);
                numberOfLosses += 1;
            }

            @Override
            public String toString() {
                return "MEAN_SQUARED_ERROR(" + losses + "," + numberOfLosses + ")";
            }
        };
    }

    /**
     * Returns a new instance of the Median cost function.
     * The Median cost function is less common in neural networks compared to MSE,
     * but can be useful for certain problems, such as outlier detection.
     * The Median cost function calculates the median absolute deviation (MAD) between the predicted output
     * and the true output across all training examples in the dataset.
     * @return a new instance of the Median cost function
     */
    public static CostFunction MEDIAN() {
        return new CostFunction() {
            @Override
            public void addLoss(double value) {
                losses += value;
                numberOfLosses += 1;
            }

            @Override
            public String toString() {
                return "MEDIAN(" + losses + "," + numberOfLosses + ")";
            }
        };
    }

    /**
     * Initializes the cost function object with the given loss and number of losses.
     *
     * @param losses         the loss value
     * @param numberOfLosses the number of losses
     * @return the initialized cost function object
     */
    private CostFunction init(double losses, int numberOfLosses) {
        this.losses = losses;
        this.numberOfLosses = numberOfLosses;
        return this;
    }

    /**
     * Extracts the cost function values from the given string representation.
     *
     * @param values the string representation of the cost function values
     * @return a Matcher object containing the extracted values
     * @throws RuntimeException if the values cannot be extracted
     */
    private static Matcher extractCostFunctionValues(String values) {
        Matcher matcher = Pattern.compile("\\((\\d+(\\.\\d+)?),(\\d+)\\)").matcher(values);

        if (matcher.find())
            return matcher;
        else throw new RuntimeException("Couldn't extract cost function values!");
    }

    /**
     * Extracts the losses from a string representation of a cost function.
     *
     * @param function the string representation of the cost function values
     * @return the losses extracted from the input string
     * @throws RuntimeException if the losses cannot be extracted from the input string
     */
    private static double extractLosses(String function) {
        return Double.parseDouble(extractCostFunctionValues(function).group(1));
    }

    /**
     * Extracts the number of losses from a string representation of a cost function.
     *
     * @param function the string representation of the cost function values
     * @return the number of losses extracted from the input string
     * @throws RuntimeException if the number of losses cannot be extracted from the input string
     */
    private static int extractNumberOfLosses(String function) {
        return Integer.parseInt(extractCostFunctionValues(function).group(3));
    }
}