#pragma once


double LearningRateDecay(int epoch, double initialLearningRate, int initialEpoch, double decayRate);


void RandomizeWeightTensor(double* weightTensor, int rows, int cols);
void XavierInitialize(double* weightTensor, int rows, int cols);
void HeInitialize(double* weightTensor, int rows, int cols);


double Linear(double* inputTensor, int inputSize, double* weightTensor);
double ReLU(double x);
double Sigmoid(double x);
double Tanh(double x);

double LossDerivative(double prediction, double label);
double ReLUDerivative(double x);
double SigmoidDerivative(double x);
double TanhDerivative(double x);

double Loss(double prediction, double label);
double BatchLoss(double predictedTensor[][1], double labelTensor[][1], int batchSize);


void Print_Tensor(double* matrix, int rows, int cols);