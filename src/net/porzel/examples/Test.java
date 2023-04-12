package net.porzel.examples;

import net.porzel.ConvolutionalNeuralNetwork;

import java.io.File;
import java.util.Random;

public class Test {
    public static void main(String[] args) {
        double[] test = new double[100000];

        for (int i = 0; i < test.length; i++) {
            test[i] = new Random().nextDouble();
        }

        long time = System.nanoTime();

        double average = 0;

        for (int i = 0; i < test.length; i++) {
            average += test[i];
        }

        average = average / test.length;


        long currentTime = System.nanoTime() - time;
        System.out.println(currentTime);

        time = System.nanoTime();

        average = 0;
        double lastDiver = 0;

        for (int i = 0; i < test.length; i++) {
            average *= lastDiver;
            average += test[i];
            average /= test.length;
            lastDiver = test.length;
        }

        currentTime = System.nanoTime() - time;
        System.out.println(currentTime / test.length);
    }
}
