package com.teinvdlugt.neuralnetwork;

import java.text.DecimalFormat;

public class Main {

    public static void main(String[] args) {
        Network network = new Network(2, 3, 1);

        double[][] x = {{3, 5}, {5, 1}, {10, 2}};
        double[][] y = {{0.75}, {0.82}, {0.93}};
        Network.normalize(x);
        network.reduceCost(x, y);
        /*double[][] yHat = network.forward(x);
        System.out.println(arrayToString(yHat));

        double[] yHat1D = column(yHat, 0);
        System.out.println("Cost: " + Network.cost(yHat1D, y));*/
    }

    public static double[] column(double[][] matrix, int column) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = matrix[i][column];
        }
        return result;
    }

    public static String arrayToString(double[][] array) {
        StringBuilder sb = new StringBuilder();
        DecimalFormat format = new DecimalFormat("0.##");
        for (double[] anArray : array) {
            for (double aDouble : anArray) {
                sb.append(format.format(aDouble)).append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}