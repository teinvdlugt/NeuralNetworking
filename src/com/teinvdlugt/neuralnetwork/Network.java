package com.teinvdlugt.neuralnetwork;


public class Network {
    private int inputLayerSize = 2;
    private int hiddenLayerSize = 3;
    private int outputLayerSize = 1;

    private double[][] w1 = new double[inputLayerSize][hiddenLayerSize];
    private double[][] w2 = new double[hiddenLayerSize][outputLayerSize];

    public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        this();
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.outputLayerSize = outputLayerSize;
    }

    public Network() {
        randomize(w1);
        randomize(w2);
    }

    public double[][] forward(double[][] input) {
        if (input[0].length != inputLayerSize)
            throw new IllegalArgumentException("input.length has to equal Network.inputLayerSize");

        double[][] z2 = applyWeights(input, w1);
        double[][] a2 = activate(z2);
        double[][] z3 = applyWeights(a2, w2);
        return activate(z3); // yHat
    }

    public static double cost(double[] yHat, double[] y) {
        if (yHat.length != y.length)
            throw new IllegalArgumentException("yHat.length has to be equal to y.length");

        double cost = 0;
        for (int i = 0; i < yHat.length; i++) {
            cost += .5 * Math.pow(y[i] - yHat[i], 2);
        }

        return cost;
    }

    public static double[][] applyWeights(double[][] input, double[][] weights) {
        if (weights.length != input[0].length)
            throw new IllegalArgumentException("input[0].length must be equal to weights.length");
        double[][] result = new double[input.length][weights[0].length]; // new double[hiddenLayerSize][dataSize]

        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                double node = 0;
                for (int k = 0; k < input[0].length; k++) {
                    // i = which data example
                    // j = which hidden node
                    // k = which input node
                    node += weights[k][j] * input[i][k];
                }
                result[i][j] = node;
            }
        }

        return result;
    }

    public static double[][] activate(double[][] x) {
        double[][] result = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = 1d / (1 + Math.pow(Math.E, -x[i][j]));
            }
        }
        return result;
    }

    public static void randomize(double[][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                array[i][j] = Math.random();
            }
        }
    }

    public static void normalize(double[][] array, double max) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                array[i][j] = array[i][j] / max;
            }
        }
    }

    public static void normalize(double[][] array) {
        // Find max value
        double max = array[0][0];
        for (double[] m : array) {
            for (double n : m) {
                if (n > max) max = n;
            }
        }

        normalize(array, max);
    }
}
