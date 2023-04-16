package net.porzel.examples;

import net.porzel.ConvolutionalNeuralNetwork;
import net.porzel.functions.ActivationFunction;
import net.porzel.functions.LossFunction;
import net.porzel.functions.WeightInitialization;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import javax.swing.*;

public class CIFAR {
    public static void main(String[] args) throws IOException {
        double[][] images = getImages();
        double[][] labels = getLabels();

        double[][] testImages = Arrays.copyOfRange(images, 0, images.length / 100);
        double[][] testLabels = Arrays.copyOfRange(labels, 0, images.length / 100);

        ConvolutionalNeuralNetwork convolutionalNeuralNetwork = new ConvolutionalNeuralNetwork(new int[] { 3072, 3072, 3072, 3072, 10 })
                .setLearningRate(0.000002)
                .setActivationFunction(ActivationFunction.LEAKY_RELU())
                .setWeightInitializationFunction(WeightInitialization.HE())
                .setLossFunction(LossFunction.CROSS_ENTROPY_LOSS());

        //ConvolutionalNeuralNetwork convolutionalNeuralNetwork = ConvolutionalNeuralNetwork.load(new File("CIFAR.nn"));

        convolutionalNeuralNetwork.setLastLayerSoftmax(true);
        convolutionalNeuralNetwork.setNumberOfProcessThreads(10);

        convolutionalNeuralNetwork.setTrainingData(images, labels);
        convolutionalNeuralNetwork.setTestData(testImages, testLabels);

        convolutionalNeuralNetwork.train(1000);

        //convolutionalNeuralNetwork.save(new File("CIFAR.nn"));

        for (int i = 0; i < images.length; i++) {
            showImage(convertToImage(images[i]), accuracyInfo(convolutionalNeuralNetwork.propagation(images[i])));
            System.out.println(convolutionalNeuralNetwork.accuracy(new double[][] { images[i] }, new double[][] { labels[i] }));
        }
    }



    private static final int IMAGE_WIDTH = 32;
    private static final int IMAGE_HEIGHT = 32;
    private static final int NUM_CHANNELS = 3;
    private static final int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;

    private static final Path CIFAR_DATA_DIR = Paths.get("cifar-10-batches-bin");


    public static double[][] getImages() throws IOException {
        double[][] images = new double[10000][];

        for (int i = 1; i <= 5; i++) {
            Path dataFile = CIFAR_DATA_DIR.resolve("data_batch_" + i + ".bin");
            try (DataInputStream in = new DataInputStream(new FileInputStream(dataFile.toFile()))) {
                for (int j = 0; j < 10000; j++) {
                    byte[] imageDataByte = new byte[IMAGE_SIZE];
                    double[] imageDataDouble = new double[IMAGE_SIZE];
                    in.readByte();
                    in.readFully(imageDataByte);

                    for (int k = 0; k < IMAGE_SIZE; k++) {
                        imageDataDouble[k] = (double)(imageDataByte[k] + 128) / 256;
                    }

                    images[j] = imageDataDouble;
                }
            }
        }

        return images;
    }

    public static double[][] getLabels() throws IOException{
        double[][] labels = new double[10000][10];

        for (int i = 1; i <= 5; i++) {
            Path dataFile = CIFAR_DATA_DIR.resolve("data_batch_" + i + ".bin");
            try (DataInputStream in = new DataInputStream(new FileInputStream(dataFile.toFile()))) {
                for (int j = 0; j < 10000; j++) { // iterate over images in batch
                    byte[] imageData = new byte[IMAGE_SIZE];
                    byte label = in.readByte();
                    in.readFully(imageData);

                    Arrays.fill(labels[j], 0);
                    labels[j][label] = 1;
                }
            }
        }

        return labels;
    }

    private static BufferedImage convertToImage(double[] imageData) {
        BufferedImage image = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                int r = (int)(imageData[y * IMAGE_HEIGHT + x] * 256) - 128;
                int g = (int)(imageData[y * IMAGE_HEIGHT + x + IMAGE_HEIGHT * IMAGE_WIDTH] * 256) - 128;
                int b = (int)(imageData[y * IMAGE_HEIGHT + x + IMAGE_HEIGHT * IMAGE_WIDTH * 2] * 256) - 128;
                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    private static void showImage(BufferedImage image, String info) {
        JFrame frame = new JFrame();

        BufferedImage scaledImage = new BufferedImage(image.getWidth() * 2, image.getHeight() * 2, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = scaledImage.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g2d.drawImage(image, 0, 0, scaledImage.getWidth(), scaledImage.getHeight(), null);
        g2d.dispose();
        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(scaledImage, 0, 0, null);

                g.setColor(Color.BLACK);
                for (int i = 0; i < info.split(";").length; i++) {
                    g.drawString(info.split(";")[i], 100, 13 * i + 20);
                }
            }
        };
        frame.setContentPane(panel);
        frame.revalidate();
        frame.repaint();
        frame.setVisible(true);
        frame.setSize(32 * 10, 32 * 10);

        try {
            Thread.sleep(100); // wait for a short time to show each image
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

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

    public static String accuracyInfo(double[] output) {
        String[] classes = new String[] { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

        StringBuilder info = new StringBuilder();

        for (int i = 0; i < output.length; i++) {
            info.append(classes[i]).append(": ").append(Math.round(output[i] * 100)).append("%;");
        }

        return info.toString();
    }
}
