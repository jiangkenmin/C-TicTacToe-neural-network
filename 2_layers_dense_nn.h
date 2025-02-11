#pragma once


// This is a dense neural network with 3 layers, weight updated by SGD


void UpdateWeights_2Layers_NN(double* inputTensor, double* labelTensor, double learningRate);

double Forward_2Layers_NN(double* inputTensor);

void Randomized_2Layers_NN_Weight(int seed);

void Print_2Layers_NN_Weight();

void Load_2layers_NN_Weight();