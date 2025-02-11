#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// training functions
double LearningRateDecay(int epoch, double initialLearningRate, int initialEpoch, double decayRate) {
    if (initialEpoch < 20)
        return initialLearningRate;
    double k = ((1 - decayRate) * initialLearningRate) / (initialEpoch / 2.0);
    if (epoch < 3 * initialEpoch / 4 && epoch >= initialEpoch / 4)
        return initialLearningRate - k * (epoch - initialEpoch / 4.0); // decay when epoch is between 1/4 and 3/4 of initialEpoch
    else if (epoch < initialEpoch / 4)
        return initialLearningRate;
    else
        return initialLearningRate * decayRate;
}

// weight initialization functions
void RandomizeWeightTensor(double* weightTensor, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols + 1; j++)
            weightTensor[i * cols + j] = ((rand() % 200) / 1000.0) + 0.1;
}
void XavierInitialize(double* weightTensor, int rows, int cols) {
    // Xavier initialization for Sigmoid or Tanh
    double scale = sqrt(6.0 / (rows + cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weightTensor[i * cols + j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}
void HeInitialize(double* weightTensor, int rows, int cols) {
    // He initialization for ReLU
    double scale = sqrt(2.0 / cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weightTensor[i * cols + j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}

// activation functions and their derivatives
double Linear(double* inputTensor, int inputSize, double* weightTensor) {
    double sum = 0;
    for (int i = 0; i < inputSize; i++) {
        sum += inputTensor[i] * weightTensor[i];
    }
    sum += weightTensor[inputSize]; // bias
    return sum;
}
double ReLU(double x) {
    return x > 0 ? x : 0.0;
}
double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double Tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double LossDerivative(double prediction, double label) {
    return 2 * (prediction - label);
}
double ReLUDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}
double SigmoidDerivative(double x) {
    return Sigmoid(x) * (1.0 - Sigmoid(x));
}
double TanhDerivative(double x) {
    return 1.0 - Tanh(x) * Tanh(x);
}

double Loss(double prediction, double label) {
    return (prediction - label) * (prediction - label);
}
double BatchLoss(double predictedTensor[][1], double labelTensor[][1], int batchSize) {
    double sum = 0;
    for (int i = 0; i < batchSize; i++) {
        sum += Loss(predictedTensor[i][0], labelTensor[i][0]);
    }
    return sum / batchSize;
}

// other functions
void Print_Tensor(double* matrix, int rows, int cols) {
    printf("{\n");
    for (int i = 0; i < rows; i++) {
        printf("{ ");
        for (int j = 0; j < cols; j++) {
            if (matrix[i * cols + j] >= 0.0) printf(" ");
            printf("%.7f", matrix[i * cols + j]);
            if (j != cols - 1)
                printf(",");
        }
        printf(" },\n");
    }
    printf(" };\n");
}