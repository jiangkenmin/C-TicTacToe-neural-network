#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

# define inputLength 2
# define batchNum 800
// This is a dense neural network with 2 layers, weight updated by SGD
# define layer1_neuronNum 4
# define layer2_neuronNum 1

static void RandomizeWeightTensor(double* weightTensor, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols + 1; j++)
            weightTensor[i * cols + j] = ((rand() % 400) / 1000.0) + 0.1;
}

static double LearningRateDecay(int epoch, double initialLearningRate, int initialEpoch) {
    return initialLearningRate * exp(-((double)epoch / initialEpoch) * 4.60517);
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

static void UpdateWeights(
    double* inputTensor,
    double* layer1_outputTensor,
    double* layer2_outputTensor,
    double* labelTensor,
    double weightTensor_1[][inputLength + 1],
    double weightTensor_2[][layer1_neuronNum + 1],
    double learningRate) {

    double layer2_error = LossDerivative(layer2_outputTensor[0], labelTensor[0]);
    double layer2_delta = layer2_error * SigmoidDerivative(layer2_outputTensor[0]);

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
static double Forward(
    double* inputTensor,
    double weightTensor_1[][inputLength + 1],
    double weightTensor_2[][layer1_neuronNum + 1],
    double* layer1_outputTensor,
    double* layer2_outputTensor) {
    // 第一层有 layer1_neuronNum 个神经元
    for (int i = 0; i < layer1_neuronNum; i++) {
        layer1_outputTensor[i] = Linear(inputTensor, inputLength, weightTensor_1[i]);
        layer1_outputTensor[i] = ReLU(layer1_outputTensor[i]);
    }

    // 第二层有 1 个神经元
    for (int i = 0; i < 1; i++) {
        layer2_outputTensor[i] = Linear(layer1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        layer2_outputTensor[i] = Sigmoid(layer2_outputTensor[i]);
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
        double temp = sqrt(dataTensor[i][0] * dataTensor[i][0] + dataTensor[i][1] * dataTensor[i][1]);
        temp = temp < 0.6 ? 1.0 : 0.0; // 判断点是否在半径为 0.6 的圆内
        labelTensor[i][0] = temp;
        printf("%f  %f  %f\n", dataTensor[i][0], dataTensor[i][1], labelTensor[i][0]);
    }

    RandomizeWeightTensor(&weightTensor_1[0][0], layer1_neuronNum, inputLength);
    RandomizeWeightTensor(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum);

    const int batchSize = batchNum;
    const double initialLearningRate = 0.005;
    const int initialEpoch = 20000;

    for (int epoch = 1; epoch <= initialEpoch; epoch++) {
        double learningRate = LearningRateDecay(epoch, initialLearningRate, initialEpoch);
        for (int i = 0; i < batchSize; i++) {
            predictedTensor[i][0] = Forward(dataTensor[i], weightTensor_1, weightTensor_2,
                layer1_outputTensor, layer2_outputTensor);
            UpdateWeights(dataTensor[i], layer1_outputTensor, layer2_outputTensor,
                labelTensor[i], weightTensor_1, weightTensor_2, learningRate);
        }
        if (epoch < 100 || epoch % 1000 == 0) {
            double batchLoss = BatchLoss(predictedTensor, labelTensor, batchSize);
            printf("Epoch %d  BatchLoss = %f  lr = %f\n", epoch, batchLoss, learningRate);
        }
    }

    // 测试循环 1 万次的时间
    double x[2] = { 0, 0 };
    double prediction;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        x[0] = (rand() % 1000 / 1000.0);
        x[1] = (rand() % 1000 / 1000.0);
        prediction = Forward(x, weightTensor_1, weightTensor_2, layer1_outputTensor, layer2_outputTensor);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        x[0] = (rand() % 1000 / 1000.0);
        x[1] = (rand() % 1000 / 1000.0);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1 - (t3 - t2));
    printf("Time elapsed: %f nanoseconds\n", duration.count() / 10000.0);

    x[0] = 0.2;
    x[1] = 0.3;
    prediction = Forward(x, weightTensor_1, weightTensor_2, layer1_outputTensor, layer2_outputTensor);
    printf("sqrt(%f^2 + %f^2) = %f  %f\n", x[0], x[1], sqrt(x[0] * x[0] + x[1] * x[1]), prediction);

    return;
}
