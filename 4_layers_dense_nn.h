#pragma once


// This is a dense neural network with 4 layers, weight updated by SGD


void UpdateWeights_4Layers_NN(double* inputTensor, double* labelTensor, double learningRate);

double Forward_4Layers_NN(double* inputTensor);

void Randomized_4Layers_NN_Weight(int seed);

void Print_4Layers_NN_Weight();

void Load_4layers_NN_Weight();