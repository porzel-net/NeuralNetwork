package net.porzel.functions;

public abstract class CostFunction {
    public abstract void addLoss(double value);

    public double losses;
    public int numberOfLosses;

    public double getCost() {
        return losses / numberOfLosses;
    }

    /**

     Resolves the loss function given a string representation of the function name.
     @param function the string representation of the cost function name
     @return the corresponding cost function object
     @throws RuntimeException if the cost function name cannot be resolved
     */
    public static CostFunction resolveCostFunction(String function) {
        return switch (function) {
            case "MEAN_SQUARED_ERROR" -> CostFunction.MEAN_SQUARED_ERROR();
            case "MEDIAN" -> CostFunction.MEDIAN();
            case "NONE" -> null;
            default -> throw new RuntimeException("Cost function couldn't be resolved!");
        };
    }


    public static CostFunction MEAN_SQUARED_ERROR() {
        return new CostFunction() {
            @Override
            public void addLoss(double value) {
                losses += Math.pow(value, 2);

                numberOfLosses += 1;
            }

            @Override
            public String toString() {
                return "MEAN_SQUARED_ERROR";
            }
        };
    }


    public static CostFunction MEDIAN() {
        return new CostFunction() {
            @Override
            public void addLoss(double value) {
                losses += value;
                numberOfLosses += 1;
            }

            @Override
            public String toString() {
                return "MEDIAN";
            }
        };
    }
}
