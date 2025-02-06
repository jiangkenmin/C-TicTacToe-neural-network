#include <chrono>
#include <iostream>
#include <stdlib.h>

# define inputLength 2
# define batchNum 100
// This is a dense neural network with 2 layers, weight updated by SGD
# define layer1_neuronNum 8
# define layer2_neuronNum 1

static void RandomizeWeightTensor(double *weightTensor, int rows, int cols) {
    for (int i = 0; i < rows; i++) 
        for (int j = 0; j < cols + 1; j++) 
            weightTensor[i * cols + j] = (rand() % 1000 / 1000.0) - 0.5;
}

static double Linear(double* inputTensor, int inputSize, double* weightTensor) {
    double sum = 0;
    for (int i = 0; i < inputSize; i++) {
        sum += inputTensor[i] * weightTensor[i];
    }
    sum += weightTensor[inputSize]; // 偏置项
    return sum;
}
static double ReLU(double x) {
    return x > 0 ? x : 0;
}
static double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double Loss(double prediction, double label) {
    return (prediction - label) * (prediction - label);
}
static double BatchLoss(double predictedTensor[][1], double labelTensor[][1], int batchSize) {
    double sum = 0;
    for (int i = 0; i < batchSize; i++) {
        sum += Loss(predictedTensor[i][0], labelTensor[i][0]);
    }
    return sum / batchSize;
}

static double LossDerivative(double prediction, double label) {
    return 2 * (prediction - label);
}
static double ReLUDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}
static double SigmoidDerivative(double x) {
    return x * (1.0 - x);
}

static void UpdateWeights(double* inputTensor, double* layer1_outputTensor, double* layer2_outputTensor, double* labelTensor, double weightTensor_1[][inputLength + 1], double weightTensor_2[][layer1_neuronNum + 1], double learningRate) {
    double layer2_error = LossDerivative(layer2_outputTensor[0], labelTensor[0]);
    //double layer2_delta = layer2_error * SigmoidDerivative(layer2_outputTensor[0]);
    double layer2_delta = layer2_error;
    // 更新第二层权重
    for (int j = 0; j < layer1_neuronNum; j++) {
        weightTensor_2[0][j] -= learningRate * layer2_delta * layer1_outputTensor[j];
    }
    weightTensor_2[0][layer1_neuronNum] -= learningRate * layer2_delta; // 更新第二层偏置

    // 计算第一层误差
    for (int j = 0; j < layer1_neuronNum; j++) {
        double layer1_error = layer2_delta * weightTensor_2[0][j];
        double layer1_delta = layer1_error * ReLUDerivative(layer1_outputTensor[j]);

        // 更新第一层权重
        for (int k = 0; k < inputLength; k++) {
            weightTensor_1[j][k] -= learningRate * layer1_delta * inputTensor[k];
        }
        weightTensor_1[j][inputLength] -= learningRate * layer1_delta; // 更新第一层偏置
    }
}
static double Forward(double* inputTensor, double weightTensor_1[][inputLength + 1], double weightTensor_2[][layer1_neuronNum + 1], double* layer1_outputTensor, double* layer2_outputTensor) {
    // 第一层有 layer1_neuronNum 个神经元
    for (int i = 0; i < layer1_neuronNum; i++) {
        layer1_outputTensor[i] = Linear(inputTensor, inputLength, weightTensor_1[i]);
        layer1_outputTensor[i] = ReLU(layer1_outputTensor[i]);
    }

    // 第二层有 1 个神经元
    for (int i = 0; i < 1; i++) {
        layer2_outputTensor[i] = Linear(layer1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        //layer2_outputTensor[i] = Sigmoid(layer2_outputTensor[i]);
    }
    return layer2_outputTensor[0];
}

double dataTensor[batchNum][inputLength] = { 0 }; // 数据集
double labelTensor[batchNum][layer2_neuronNum] = { 0 }; // 标签集
double weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // 第一层权重
double weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // 第二层权重

double layer1_outputTensor[layer1_neuronNum] = { 0 };
double layer2_outputTensor[layer2_neuronNum] = { 0 };
double predictedTensor[batchNum][1] = { 0 };

void main() {
    // 生成数据集
    for (int i = 0; i < batchNum; i++) {
        dataTensor[i][0] = rand() % 1000 / 1000.0;
        dataTensor[i][1] = rand() % 1000 / 1000.0;
        double temp = dataTensor[i][0] * 2.0 + dataTensor[i][1];
        labelTensor[i][0] = temp;
        std::cout << "dataTensor[" << i << "][0] = " << dataTensor[i][0] << ", dataTensor[" << i << "][1] = " << dataTensor[i][1] 
            << ", labelTensor[" << i << "] = " << labelTensor[i][0] << std::endl;
    }

    RandomizeWeightTensor(&weightTensor_1[0][0], layer1_neuronNum, inputLength);
    RandomizeWeightTensor(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum);

    int batchSize = batchNum;
    double learningRate = 0.001;

    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int i = 0; i < batchSize; i++) {
            predictedTensor[i][0] = Forward(dataTensor[i], inputLength, weightTensor_1, weightTensor_2, layer1_outputTensor, layer2_outputTensor);
            UpdateWeights(dataTensor[i], layer1_outputTensor, layer2_outputTensor, labelTensor[i], weightTensor_1, weightTensor_2, learningRate);
        }

        double loss = BatchLoss(predictedTensor, labelTensor, batchSize);
        std::cout << "Epoch" << epoch << ": loss = " << loss << std::endl;
    }

    // 测试循环 10 万次的时间
    double x[2] = { 0, 0 };
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        x[0] = (rand() % 1000 / 1000.0) - 0.5;
        x[1] = (rand() % 1000 / 1000.0) - 0.5;
        double prediction = Forward(x, inputLength, weightTensor_1, weightTensor_2, layer1_outputTensor, layer2_outputTensor);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time elapsed: " << duration.count() << " microseconds\n";

    x[0] = 0.3;
    x[1] = 0.2;
    double prediction = Forward(x, weightTensor_1, weightTensor_2, layer1_outputTensor, layer2_outputTensor);
    std::cout << x[0] << " " << x[1] << " " << prediction << std::endl;

    return;
}
