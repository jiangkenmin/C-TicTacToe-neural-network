#pragma once


// This is a dense neural network with 3 layers, weight updated by SGD


void UpdateWeights_3Layers_NN(double* inputTensor, double* labelTensor, double learningRate);

double Forward_3Layers_NN(double* inputTensor);

void Randomized_3Layers_NN_Weight(int seed);

void Print_3Layers_NN_Weight();

void Load_3layers_NN_Weight();