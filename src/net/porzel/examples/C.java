package net.porzel.examples;

import net.porzel.ConvolutionalNeuralNetwork;
import net.porzel.functions.ActivationFunction;
import net.porzel.functions.WeightInitialization;


import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class C {
    public static void main(String[] args) {
        DrawApp drawApp = new DrawApp();
    }


    public static class DrawApp extends JFrame implements MouseListener, MouseMotionListener {
        private static final int IMAGE_WIDTH = 28;
        private static final int IMAGE_HEIGHT = 28;
        private static final int PIXEL_SIZE = 20;
        private int[][] pixels = new int[IMAGE_WIDTH][IMAGE_HEIGHT];

        private JPanel canvas;
        private Graphics graphics;
        private boolean mousePressed = false;

        public DrawApp() {
            ConvolutionalNeuralNetwork neuralNetwork = ConvolutionalNeuralNetwork.load(new File("number-recognition-sigmoid-6.nn"));

            setTitle("Draw App");
            setSize(IMAGE_WIDTH * PIXEL_SIZE, IMAGE_HEIGHT * PIXEL_SIZE);
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            for (int x = 0; x < pixels.length; x++) {
                Arrays.fill(pixels[x], 0);
            }

            canvas = new JPanel() {
                @Override
                protected void paintComponent(Graphics g) {
                    double[] imageLinear = new double[IMAGE_WIDTH * IMAGE_HEIGHT];

                    for (int x = 0; x < pixels.length; x++) {
                        for (int y = 0; y < pixels[x].length; y++) {
                            if(pixels[x][y] == 0)
                                g.setColor(Color.BLACK);
                            else
                                g.setColor(Color.WHITE);

                            g.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);

                            imageLinear[x * pixels.length + y] = pixels[y][x];
                        }
                    }

                    g.setColor(Color.WHITE);

                    double[] predicted = neuralNetwork.propagation(imageLinear);

                    double[] accuracy = new double[10];

                    for (int i = 0; i < accuracy.length; i++) {
                        accuracy[i] = Math.round(accuracy(predicted, getBitArray(i)) * 100);
                    }

                    int height = 1;

                    while (true) {
                        int index = -1;

                        for (int i = 0; i < accuracy.length; i++) {
                            if(accuracy[i] != 101 && index == -1) {
                                index = i;
                            }
                            else if(accuracy[i] != 101 && accuracy[index] < accuracy[i]) {
                                index = i;
                            }
                        }

                        if(index == -1)
                            break;

                        g.drawString(index + ": " + accuracy[index] + "%", 20, 20 * height);

                        accuracy[index] = 101;

                        height += 1;
                    }
                }
            };
            canvas.setPreferredSize(new Dimension(IMAGE_WIDTH * PIXEL_SIZE, IMAGE_HEIGHT * PIXEL_SIZE));
            canvas.addMouseListener(this);
            canvas.addMouseMotionListener(this);

            getContentPane().add(canvas, BorderLayout.CENTER);

            setVisible(true);
            graphics = canvas.getGraphics();
            clearCanvas();
        }

        private void clearCanvas() {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, IMAGE_WIDTH * PIXEL_SIZE, IMAGE_HEIGHT * PIXEL_SIZE);
            for (int i = 0; i < IMAGE_WIDTH; i++) {
                for (int j = 0; j < IMAGE_HEIGHT; j++) {
                    pixels[i][j] = 0;
                }
            }

            repaint();
        }

        private void drawPixel(int x, int y) {
            graphics.setColor(Color.BLACK);
            pixels[x][y] = 1;

            pixels[x - 1][y] = 1;
            pixels[x + 1][y] = 1;
            pixels[x][y - 1] = 1;
            pixels[x][y + 1] = 1;

            canvas.repaint();
        }

        private void saveImage() {
            BufferedImage image = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
            for (int i = 0; i < IMAGE_WIDTH; i++) {
                for (int j = 0; j < IMAGE_HEIGHT; j++) {
                    int pixelValue = pixels[i][j] == 1 ? 0 : 255; // Invert the pixel value because the canvas draws black on white
                    image.setRGB(i, j, (pixelValue << 16) | (pixelValue << 8) | pixelValue);
                }
            }

            // Save the image as JPG file
            File output = new File("image.jpg");
            try {
                ImageIO.write(image, "jpg", output);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            // not used
        }

        @Override
        public void mousePressed(MouseEvent e) {
            mousePressed = true;
            int x = e.getX() / PIXEL_SIZE;
            int y = e.getY() / PIXEL_SIZE;
            drawPixel(x, y);
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            mousePressed = false;
            saveImage();
        }

        @Override
        public void mouseEntered(MouseEvent e) {
            // not used
        }

        @Override
        public void mouseExited(MouseEvent e) {
            clearCanvas();
        }

        @Override
        public void mouseDragged(MouseEvent e) {
            if (mousePressed) {
                int x = e.getX() / PIXEL_SIZE;
                int y = e.getY() / PIXEL_SIZE;
                drawPixel(x, y);
            }
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            // not used
        }

        @Override
        public void paintComponents(Graphics g) {
            super.paintComponents(g);
        }
    }

    public static double accuracy(double[] output, double[] target) {
        double error = 0;

        for (int j = 0; j < output.length; j++) {
            error += Math.abs(output[j] - target[j]);
        }

        return 1 - (error / output.length);
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
}
