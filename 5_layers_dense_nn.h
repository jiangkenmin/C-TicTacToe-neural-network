#pragma once


// This is a dense neural network with 4 layers, weight updated by SGD


void UpdateWeights_5Layers_NN(double* inputTensor, double* labelTensor, double learningRate);

double Forward_5Layers_NN(double* inputTensor);

void Randomized_5Layers_NN_Weight(int seed);

void Print_5Layers_NN_Weight();

void Load_5layers_NN_Weight();