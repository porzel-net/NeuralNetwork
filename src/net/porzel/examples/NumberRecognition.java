package net.porzel.examples;

import net.porzel.ConvolutionalNeuralNetwork;
import net.porzel.functions.ActivationFunction;
import net.porzel.functions.CostFunction;
import net.porzel.functions.LossFunction;
import net.porzel.functions.WeightInitialization;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class NumberRecognition {

    public static double[][] loadImages(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        byte[] data = Files.readAllBytes(path);

        int magicNumber = ByteBuffer.wrap(data, 0, 4).order(ByteOrder.BIG_ENDIAN).getInt();
        if (magicNumber != 2051) {
            throw new IOException("Invalid magic number: " + magicNumber);
        }

        int numImages = ByteBuffer.wrap(data, 4, 4).order(ByteOrder.BIG_ENDIAN).getInt();
        int numRows = ByteBuffer.wrap(data, 8, 4).order(ByteOrder.BIG_ENDIAN).getInt();
        int numCols = ByteBuffer.wrap(data, 12, 4).order(ByteOrder.BIG_ENDIAN).getInt();

        double[][] images = new double[numImages][numRows * numCols];

        for (int i = 0; i < numImages; i++) {
            for (int j = 0; j < numRows * numCols; j++) {
                images[i][j] = (double) (data[16 + i * numRows * numCols + j] & 0xff) / 255.0;
            }
        }

        return images;
    }

    public static double[][] loadLabels(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        byte[] data = Files.readAllBytes(path);

        int magicNumber = ByteBuffer.wrap(data, 0, 4).order(ByteOrder.BIG_ENDIAN).getInt();
        if (magicNumber != 2049) {
            throw new IOException("Invalid magic number: " + magicNumber);
        }

        int numLabels = ByteBuffer.wrap(data, 4, 4).order(ByteOrder.BIG_ENDIAN).getInt();

        double[][] labels = new double[numLabels][10];

        for (int i = 0; i < numLabels; i++) {
            int label = data[8 + i] & 0xff;

            labels[i] = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            labels[i][label] = 1;
        }

        return labels;
    }



    public static void main(String[] args) throws IOException {
        double[][] trainImages = loadImages(System.getProperty("user.dir") + "/train-images.idx3-ubyte");
        double[][] trainLabels = loadLabels(System.getProperty("user.dir") + "/train-labels.idx1-ubyte");


        double[][] testImages = Arrays.copyOfRange(loadImages(System.getProperty("user.dir") + "/t10k-images.idx3-ubyte"), 0, 10000 / 100);
        double[][] testLabels = Arrays.copyOfRange(loadLabels(System.getProperty("user.dir") + "/t10k-labels.idx1-ubyte"), 0, 10000 / 100);


        /*ConvolutionalNeuralNetwork convolutionalNeuralNetwork = new ConvolutionalNeuralNetwork(new int[] {784, 1568, 784, 784, 392, 392, 10})
                .setLearningRate(0.00015)
                .setActivationFunction(ActivationFunction.LEAKY_RELU())
                .setWeightInitializationFunction(WeightInitialization.HE())
                .setLossFunction(LossFunction.CROSS_ENTROPY_LOSS())
                .setCostFunction(CostFunction.MEDIAN());*/

        ConvolutionalNeuralNetwork convolutionalNeuralNetwork = ConvolutionalNeuralNetwork.load(new File("numberRecognition-0.nn"));
        for (int i = 0; i < 10; i++) {

            for (int j = 0; j < testImages.length; j++) {
                ImageDrawer.showImage(testImages[j], accuracyInfo(convolutionalNeuralNetwork.propagation(testImages[j])));
            }

            //convolutionalNeuralNetwork.save(new File("numberRecognition-" + i + ".nn"));
        }


        convolutionalNeuralNetwork.setLastLayerSoftmax(true);
        convolutionalNeuralNetwork.setNumberOfProcessThreads(10);
        convolutionalNeuralNetwork.setTrainingData(trainImages, trainLabels);
        convolutionalNeuralNetwork.setTestData(testImages, testLabels);

        for (int i = 0; i < 10; i++) {
            convolutionalNeuralNetwork.train(10000);

            for (int j = 0; j < 10; j++) {
                ImageDrawer.showImage(testImages[j], accuracyInfo(convolutionalNeuralNetwork.propagation(testImages[j])));
            }

            convolutionalNeuralNetwork.save(new File("numberRecognition-" + i + ".nn"));
        }
    }

    public static String accuracyInfo(double[] output) {
        StringBuilder info = new StringBuilder();

        for (int i = 0; i < output.length; i++) {
            info.append(i).append(": ").append(Math.round(output[i] * 100)).append("%;");
        }

        return info.toString();
    }


    static class ImageDrawer extends JPanel {

        private double[] image;
        private int imageSize;
        private String info;

        public ImageDrawer(double[] image, String info) {
            this.image = image;
            this.imageSize = (int)Math.sqrt(image.length);

            // Set the size of the panel
            setPreferredSize(new java.awt.Dimension(imageSize * 20, imageSize * 20));

            this.info = info;
        }

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);

            // Draw the pixels of the image on the panel
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int pixelValue = (int) (image[i * imageSize + j] * 255);
                    g.setColor(new Color(pixelValue, pixelValue, pixelValue));
                    g.fillRect(j * 20, i * 20, 1 * 20, 1 * 20);
                }
            }

            g.setColor(Color.white);
            for (int i = 0; i < info.split(";").length; i++) {
                g.drawString(info.split(";")[i], 10, 20 * i + 20);
            }
        }

        public static void showImage(double[] image, String info) {
            JFrame frame = new JFrame("Image");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new ImageDrawer(image, info));
            frame.pack();
            frame.setVisible(true);

            frame.addKeyListener(new KeyListener() {
                @Override
                public void keyTyped(KeyEvent e) {

                }

                @Override
                public void keyPressed(KeyEvent e) {
                    if(e.getKeyCode() == KeyEvent.VK_ENTER) {
                        frame.dispose();
                        frame.setVisible(false);
                    }
                }

                @Override
                public void keyReleased(KeyEvent e) {

                }
            });

            while (frame.isVisible()) {
                System.out.print("");
            }
        }
    }
}
