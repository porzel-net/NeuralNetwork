package net.porzel.examples;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;


import net.porzel.ConvolutionalNeuralNetwork;
import net.porzel.functions.ActivationFunction;
import net.porzel.functions.CostFunction;
import net.porzel.functions.LossFunction;
import net.porzel.functions.WeightInitialization;

public class MNISTReader {

    private static final String IMAGE_FILE_MAGIC_NUMBER = "00000803";
    private static final String LABEL_FILE_MAGIC_NUMBER = "00000801";
    private static final int IMAGES_HEADER_SIZE = 16;
    private static final int LABELS_HEADER_SIZE = 8;

    public static double[][] readImages(String filePath) throws IOException {
        FileInputStream inputStream = new FileInputStream(filePath);

        // Read the magic number
        byte[] magicNumberBytes = new byte[4];
        inputStream.read(magicNumberBytes);
        String magicNumber = bytesToHex(magicNumberBytes);
        if (!magicNumber.equals(IMAGE_FILE_MAGIC_NUMBER)) {
            throw new IOException("Invalid magic number in MNIST image file");
        }

        // Read the number of images
        byte[] numImagesBytes = new byte[4];
        inputStream.read(numImagesBytes);
        int numImages = byteArrayToInt(numImagesBytes);

        // Read the number of rows and columns per image
        byte[] numRowsBytes = new byte[4];
        byte[] numColsBytes = new byte[4];
        inputStream.read(numRowsBytes);
        inputStream.read(numColsBytes);
        int numRows = byteArrayToInt(numRowsBytes);
        int numCols = byteArrayToInt(numColsBytes);

        // Read the image data
        double[][] images = new double[numImages][numRows * numCols];
        for (int i = 0; i < numImages; i++) {
            byte[] pixelValues = new byte[numRows * numCols];
            inputStream.read(pixelValues);
            for (int j = 0; j < pixelValues.length; j++) {
                images[i][j] = (double) (pixelValues[j] & 0xFF) / 255.0;
            }
        }

        inputStream.close();
        return images;
    }

    public static double[][] readLabels(String filePath) throws IOException {
        FileInputStream inputStream = new FileInputStream(filePath);

        // Read the magic number
        byte[] magicNumberBytes = new byte[4];
        inputStream.read(magicNumberBytes);
        String magicNumber = bytesToHex(magicNumberBytes);
        if (!magicNumber.equals(LABEL_FILE_MAGIC_NUMBER)) {
            throw new IOException("Invalid magic number in MNIST label file");
        }

        // Read the number of labels
        byte[] numLabelsBytes = new byte[4];
        inputStream.read(numLabelsBytes);
        int numLabels = byteArrayToInt(numLabelsBytes);

        // Read the label data
        double[][] labels = new double[numLabels][4];
        for (int i = 0; i < numLabels; i++) {
            byte[] labelValue = new byte[1];
            inputStream.read(labelValue);

            // Convert the label to a 4-byte binary value
            String binaryLabel = String.format("%8s", Integer.toBinaryString(labelValue[0] & 0xFF)).replace(' ', '0').substring(4, 8);
            for (int j = 0; j < 4; j++) {
                labels[i][j] = Double.parseDouble(String.valueOf(binaryLabel.charAt(j)));
            }
        }

        inputStream.close();
        return labels;
    }


    private static String bytesToHex(byte[] bytes) {
        StringBuilder hexString = new StringBuilder();
        for (byte b : bytes) {
            hexString.append(String.format("%02X", b));
        }
        return hexString.toString();
    }

    private static int byteArrayToInt(byte[] bytes) {
        int value = 0;
        for (int i = 0; i < bytes.length; i++) {
            value <<= 8;
            value |= (bytes[i] & 0xFF);
        }
        return value;
    }

    public static void main(String[] args) {
        try {
            // Load images and labels

            String imagesFilePath = System.getProperty("user.dir") + "/train-images.idx3-ubyte";
            String labelsFilePath = System.getProperty("user.dir") + "/train-labels.idx1-ubyte";

            String testImagesFilePath = System.getProperty("user.dir") + "/t10k-images.idx3-ubyte";
            String testLabelsFilePath = System.getProperty("user.dir") + "/t10k-labels.idx1-ubyte";

            double[][] images = readImages(imagesFilePath);
            double[][] labels = readLabels(labelsFilePath);

            double[][] testImages = readImages(testImagesFilePath);
            double[][] testLabels = readLabels(testLabelsFilePath);

            testImages = Arrays.copyOfRange(testImages, 0, testImages.length / 10);
            testLabels = Arrays.copyOfRange(testLabels, 0, testLabels.length / 10);

            ConvolutionalNeuralNetwork neuralNetwork = new ConvolutionalNeuralNetwork(new int[] {784, 784, 784, 784, 784, 784, 784, 4})
                    .setLearningRate(0.0000001)
                    .setActivationFunction(ActivationFunction.LEAKY_RELU())
                    .setWeightInitializationFunction(WeightInitialization.HE())
                    .setLossFunction(LossFunction.BINARY_CROSS_ENTROPY_LOSS())
                    .setCostFunction(CostFunction.MEDIAN());

            //optimal war 0.002 oder 0.0025 bei SIGMOID
            //NeuralNetwork neuralNetwork = NeuralNetwork.load(new File("number-recognition-sigmoid-5.nn"));

            neuralNetwork.setTrainingData(images, labels);

            neuralNetwork.setTestData(testImages, testLabels);

            neuralNetwork.setDropout(0.0d);

            neuralNetwork.setNumberOfProcessThreads(7);

            neuralNetwork.train(1000);

            //neuralNetwork.save(new File("number-recognition-sigmoid-6.nn"));

            for (int i = 0; i < images.length; i++) {
                double[] output = neuralNetwork.propagation(testImages[i]);

                String string = "";

                for (int j = 0; j < 10; j++) {
                    string += j + ": " + Math.round(accuracy(output, getBitArray(j)) * 100) + "%;";
                }

                ImageDrawer.showImage(testImages[i], 28, string);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double[] getBitArray(int num) {
        if (num < 0 || num > 9) {
            throw new IllegalArgumentException("Number must be between 0 and 9");
        }

        double[] bitArray = new double[4];
        int mask = 0b1000;
        for (int i = 0; i < bitArray.length; i++) {
            if ((num & mask) == mask) {
                bitArray[i] = 1.0;
            } else {
                bitArray[i] = 0.0;
            }
            mask >>= 1;
        }

        return bitArray;
    }

    public static double accuracy(double[] output, double[] target) {
        double error = 0;

        for (int j = 0; j < output.length; j++) {
            error += Math.abs(output[j] - target[j]);
        }

        return 1 - (error / output.length);
    }


    static class ImageDrawer extends JPanel {

        private double[] image;
        private int imageSize;
        private String info;

        public ImageDrawer(double[] image, int imageSize, String info) {
            this.image = image;
            this.imageSize = imageSize;

            // Set the size of the panel
            setPreferredSize(new Dimension(imageSize * 20, imageSize * 20));

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

        public static void showImage(double[] image, int imageSize, String info) {
            JFrame frame = new JFrame("Image");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new ImageDrawer(image, imageSize, info));
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
