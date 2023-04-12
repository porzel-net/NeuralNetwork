package net.porzel.examples;

import java.util.ArrayList;
import java.util.List;

public class Driver {

    static double [][] X= {
            {0,0},
            {1,0},
            {0,1},
            {1,1}
    };
    static double [][] Y= {
            {0},{1},{1},{0}
    };

    public static void main(String[] args) {

        NeuralNetwork nn = new NeuralNetwork(2,3,1);


        List<Double>output;

        nn.fit(X, Y, 500000);

        double [][] input = {
                {0,0},{1,0},{0,1},{1,1}
        };

        for(double d[]:input) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }
    }


    static class Matrix {
        double [][]data;
        int rows,cols;

        public Matrix(int rows,int cols) {
            data= new double[rows][cols];
            this.rows=rows;
            this.cols=cols;
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                {
                    //data[i][j]=Math.random()*2-1;
                    data[i][j]= (i * 2 + 1 - 2.6) / 2;
                }
            }
        }

        public void print()
        {
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                {
                    System.out.print(this.data[i][j]+" ");
                }
                System.out.println();
            }
        }

        public void add(int scaler)
        {
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                {
                    this.data[i][j]+=scaler;
                }

            }
        }

        public void add(Matrix m)
        {
            if(cols!=m.cols || rows!=m.rows) {
                System.out.println("Shape Mismatch");
                return;
            }

            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                {
                    this.data[i][j]+=m.data[i][j];
                }
            }
        }

        public static Matrix fromArray(double[]x)
        {
            Matrix temp = new Matrix(x.length,1);
            for(int i =0;i<x.length;i++)
                temp.data[i][0]=x[i];
            return temp;

        }

        public List<Double> toArray() {
            List<Double> temp= new ArrayList<Double>()  ;

            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                {
                    temp.add(data[i][j]);
                }
            }
            return temp;
        }

        public static Matrix subtract(Matrix a, Matrix b) {
            Matrix temp = new Matrix(a.rows,a.cols);
            for(int i=0;i<a.rows;i++)
            {
                for(int j=0;j<a.cols;j++)
                {
                    temp.data[i][j]=a.data[i][j]-b.data[i][j];
                }
            }
            return temp;
        }

        public static Matrix transpose(Matrix a) {
            Matrix temp=new Matrix(a.cols,a.rows);
            for(int i=0;i<a.rows;i++)
            {
                for(int j=0;j<a.cols;j++)
                {
                    temp.data[j][i]=a.data[i][j];
                }
            }
            return temp;
        }

        public static Matrix multiply(Matrix a, Matrix b) {
            Matrix temp=new Matrix(a.rows,b.cols);
            for(int i=0;i<temp.rows;i++) {
                for(int j=0;j<temp.cols;j++) {
                    double sum=0;
                    for(int k=0;k<a.cols;k++) {
                        sum+=a.data[i][k]*b.data[k][j];
                    }
                    temp.data[i][j]=sum;
                }
            }
            return temp;
        }

        public void multiply(Matrix a) {
            for(int i=0;i<a.rows;i++)
            {
                for(int j=0;j<a.cols;j++)
                {
                    this.data[i][j]*=a.data[i][j];
                }
            }

        }

        public void multiply(double a) {
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                {
                    this.data[i][j]*=a;
                }
            }

        }

        public void sigmoid() {
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                    this.data[i][j] = 1/(1+Math.exp(-this.data[i][j]));
            }

        }

        public Matrix dsigmoid() {
            Matrix temp=new Matrix(rows,cols);
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<cols;j++)
                    temp.data[i][j] = this.data[i][j] * (1-this.data[i][j]);
            }
            return temp;

        }
    }

    static class NeuralNetwork {

        Matrix weights_ih,weights_ho , bias_h,bias_o;
        double l_rate = 0.01;

        public NeuralNetwork(int i,int h,int o) {
            weights_ih = new Matrix(h,i);
            weights_ho = new Matrix(o,h);

            bias_h = new Matrix(h,1);
            bias_o = new Matrix(o,1);
        }

        public List<Double> predict(double[] X)
        {
            Matrix input = Matrix.fromArray(X);
            Matrix hidden = Matrix.multiply(weights_ih, input);
            hidden.add(bias_h);
            hidden.sigmoid();

            Matrix output = Matrix.multiply(weights_ho,hidden);
            output.add(bias_o);
            output.sigmoid();

            return output.toArray();
        }


        public void fit(double[][]X,double[][]Y,int epochs) {
            for(int i=0;i<epochs;i++) {
                //int sampleN =  (int)(Math.random() * X.length );
                int sampleN = i % X.length;
                this.train(X[sampleN], Y[sampleN]);
            }
        }

        public void train(double [] X,double [] Y)
        {
            //PROPAGATION
            Matrix input = Matrix.fromArray(X);
            Matrix hidden = Matrix.multiply(weights_ih, input);
            hidden.add(bias_h);
            hidden.sigmoid();

            Matrix output = Matrix.multiply(weights_ho,hidden);
            output.add(bias_o);
            output.sigmoid();

            //BACKPROPAGATION
            Matrix target = Matrix.fromArray(Y);

            //OUTPUT LAYER
            Matrix error = Matrix.subtract(target, output);
            Matrix gradient = output.dsigmoid();
            gradient.multiply(error);
            gradient.multiply(l_rate);

            Matrix hidden_T = Matrix.transpose(hidden);
            Matrix who_delta =  Matrix.multiply(gradient, hidden_T);

            weights_ho.add(who_delta);
            bias_o.add(gradient);

            //HIDDEN LAYER
            Matrix who_T = Matrix.transpose(weights_ho);
            Matrix hidden_errors = Matrix.multiply(who_T, error);

            Matrix h_gradient = hidden.dsigmoid();
            h_gradient.multiply(hidden_errors);
            h_gradient.multiply(l_rate);

            Matrix i_T = Matrix.transpose(input);
            Matrix wih_delta = Matrix.multiply(h_gradient, i_T);

            weights_ih.add(wih_delta);
            bias_h.add(h_gradient);

            System.out.print("");
        }
    }
}
