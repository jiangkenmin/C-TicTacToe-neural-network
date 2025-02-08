#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

# define inputLength 2
# define batchNum 1200
// This is a dense neural network with 3 layers, weight updated by SGD
# define layer1_neuronNum 128   // ?
# define layer2_neuronNum 8     // ?
# define layer3_neuronNum 8     // ?
# define layer4_neuronNum 1     // ?

static void RandomizeWeightTensor(double* weightTensor, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols + 1; j++)
            weightTensor[i * cols + j] = ((rand() % 200) / 1000.0) + 0.1;
}
static void XavierInitialize(double* weightTensor, int rows, int cols) {
    // Xavier initialization for Sigmoid or Tanh
    double scale = sqrt(6.0 / (rows + cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weightTensor[i * cols + j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}
static void HeInitialize(double* weightTensor, int rows, int cols) {
    // He initialization for ReLU
    double scale = sqrt(2.0 / cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weightTensor[i * cols + j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}

static double LearningRateDecay(int epoch, double initialLearningRate, int initialEpoch) {
    if (epoch < initialEpoch / 2)
        return initialLearningRate * exp(-((double)epoch / (initialEpoch / 2)) * 4.60517);
    else
        return initialLearningRate * 0.01;
}

static double Linear(double* inputTensor, int inputSize, double* weightTensor) {
    double sum = 0;
    for (int i = 0; i < inputSize; i++) {
        sum += inputTensor[i] * weightTensor[i];
    }
    sum += weightTensor[inputSize]; // ƫ����
    return sum;
}
static double ReLU(double x) {
    return x > 0 ? x : 0.0;
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
    return Sigmoid(x) * (1.0 - Sigmoid(x));
}
static void UpdateWeights(
    double* inputTensor,
    double* layer1_outputTensor,
    double* layer2_outputTensor,
    double* layer3_outputTensor,
    double* layer4_outputTensor,
    double* labelTensor,
    double weightTensor_1[][inputLength + 1],
    double weightTensor_2[][layer1_neuronNum + 1],
    double weightTensor_3[][layer2_neuronNum + 1],
    double weightTensor_4[][layer3_neuronNum + 1],
    double learningRate) {

    // ������Ĳ����
    double layer4_error = LossDerivative(layer4_outputTensor[0], labelTensor[0]);
    layer4_error *= SigmoidDerivative(layer4_outputTensor[0]);
    // ������������
    double layer3_error[layer3_neuronNum];
    for (int j = 0; j < layer3_neuronNum; j++) {
        layer3_error[j] = layer4_error * weightTensor_4[0][j];
        layer3_error[j] *= ReLUDerivative(layer3_outputTensor[j]);
    }
    // ����ڶ������
    double layer2_error[layer2_neuronNum];
    for (int j = 0; j < layer2_neuronNum; j++) {
        layer2_error[j] = 0;
        for (int k = 0; k < layer3_neuronNum; k++) {
            layer2_error[j] += layer3_error[k] * weightTensor_3[k][j];
        }
        layer2_error[j] *= ReLUDerivative(layer2_outputTensor[j]);
    }
    // �����һ�����
    double layer1_error[layer1_neuronNum];
    for (int j = 0; j < layer1_neuronNum; j++) {
        layer1_error[j] = 0;
        for (int k = 0; k < layer2_neuronNum; k++) {
            layer1_error[j] += layer2_error[k] * weightTensor_2[k][j];
        }
        layer1_error[j] *= ReLUDerivative(layer1_outputTensor[j]);
    }

    // ���µ��Ĳ�Ȩ�غ�ƫ��
    for (int j = 0; j < layer3_neuronNum; j++) {
        weightTensor_4[0][j] -= learningRate * layer4_error * layer3_outputTensor[j];
    }
    weightTensor_4[0][layer3_neuronNum] -= learningRate * layer4_error; // ���µ��Ĳ�ƫ��
    // ���µ�����Ȩ�غ�ƫ��
    for (int j = 0; j < layer3_neuronNum; j++) {
        for (int k = 0; k < layer2_neuronNum; k++) {
            weightTensor_3[j][k] -= learningRate * layer3_error[j] * layer2_outputTensor[k];
        }
        weightTensor_3[j][layer2_neuronNum] -= learningRate * layer3_error[j]; // ���µ�����ƫ��
    }
    // ���µڶ���Ȩ�غ�ƫ��
    for (int j = 0; j < layer2_neuronNum; j++) {
        for (int k = 0; k < layer1_neuronNum; k++) {
            weightTensor_2[j][k] -= learningRate * layer2_error[j] * layer1_outputTensor[k];
        }
        weightTensor_2[j][layer1_neuronNum] -= learningRate * layer2_error[j]; // ���µڶ���ƫ��
    }
    // ���µ�һ��Ȩ�غ�ƫ��
    for (int j = 0; j < layer1_neuronNum; j++) {

        for (int k = 0; k < inputLength; k++) {
            weightTensor_1[j][k] -= learningRate * layer1_error[j] * inputTensor[k];
        }
        weightTensor_1[j][inputLength] -= learningRate * layer1_error[j]; // ���µ�һ��ƫ��
    }
}

static double Forward(
    double* inputTensor,
    double weightTensor_1[][inputLength + 1],
    double weightTensor_2[][layer1_neuronNum + 1],
    double weightTensor_3[][layer2_neuronNum + 1],
    double weightTensor_4[][layer3_neuronNum + 1],
    double* layer1_outputTensor,
    double* layer2_outputTensor,
    double* layer3_outputTensor,
    double* layer4_outputTensor) {
    // ��һ���� layer1_neuronNum ����Ԫ
    for (int i = 0; i < layer1_neuronNum; i++) {
        layer1_outputTensor[i] = Linear(inputTensor, inputLength, weightTensor_1[i]);
        layer1_outputTensor[i] = ReLU(layer1_outputTensor[i]);
    }
    // �ڶ����� layer2_neuronNum ����Ԫ
    for (int i = 0; i < layer2_neuronNum; i++) {
        layer2_outputTensor[i] = Linear(layer1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        layer2_outputTensor[i] = ReLU(layer2_outputTensor[i]);
    }
    // �������� layer3_neuronNum ����Ԫ
    for (int i = 0; i < layer3_neuronNum; i++) {
        layer3_outputTensor[i] = Linear(layer2_outputTensor, layer2_neuronNum, weightTensor_3[i]);
        layer3_outputTensor[i] = ReLU(layer3_outputTensor[i]);
    }
    // �����ֻ��һ����Ԫ
    layer4_outputTensor[0] = Linear(layer3_outputTensor, layer3_neuronNum, weightTensor_4[0]);
    layer4_outputTensor[0] = Sigmoid(layer4_outputTensor[0]);

    return layer4_outputTensor[0];
}

double dataTensor[batchNum][inputLength] = { 0 }; // ���ݼ�
double labelTensor[batchNum][layer4_neuronNum] = { 0 }; // ��ǩ��

double weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // ��һ��Ȩ�ؾ���
double weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // �ڶ���Ȩ�ؾ���
double weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] = { 0 }; // ������Ȩ�ؾ���
double weightTensor_4[layer4_neuronNum][layer3_neuronNum + 1] = { 0 }; // �����Ȩ�ؾ���

double layer1_outputTensor[layer1_neuronNum] = { 0 };
double layer2_outputTensor[layer2_neuronNum] = { 0 };
double layer3_outputTensor[layer3_neuronNum] = { 0 };
double layer4_outputTensor[layer4_neuronNum] = { 0 };

double predictedTensor[batchNum][1] = { 0 };

void main() {
    // �������ݼ�
    for (int i = 0; i < batchNum; i++) {
        dataTensor[i][0] = rand() % 1000 / 1000.0;
        dataTensor[i][1] = rand() % 1000 / 1000.0;
        double temp = dataTensor[i][0] * dataTensor[i][0] / 0.16 + dataTensor[i][1] * dataTensor[i][1] / 0.36;
        temp = temp < 1.0 ? 1.0 : 0.0; // �жϵ��Ƿ��� x ��ؾ�Ϊ 0.4��y ��ؾ�Ϊ 0.6 ����Բ��
        labelTensor[i][0] = temp;
        if (i < 10) printf("%f  %f  %f\n", dataTensor[i][0], dataTensor[i][1], labelTensor[i][0]);
    }
    // ReLU ��Ԫ�� He ��ʼ��Ȩ�أ�Sigmoid ��Ԫ�� Xavier ��ʼ��Ȩ�أ�ƫ������ 0 ��ʼ��������ʼ������������ + 1��
    XavierInitialize(&weightTensor_1[0][0], layer1_neuronNum, inputLength);
    XavierInitialize(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum);
    XavierInitialize(&weightTensor_3[0][0], layer3_neuronNum, layer2_neuronNum);
    HeInitialize(&weightTensor_4[0][0], layer4_neuronNum, layer3_neuronNum);

    const int batchSize = batchNum;
    const double initialLearningRate = 0.02;
    const int initialEpoch = 10000;
    const int patience = 1000;

    double lowestLoss = 1000000.0;
    int lowestLossEpoch = 0;
    for (int epoch = 1; epoch <= initialEpoch; epoch++) {
        double learningRate = LearningRateDecay(epoch, initialLearningRate, initialEpoch);
        for (int i = 0; i < batchSize; i++) {
            predictedTensor[i][0] = Forward(dataTensor[i], weightTensor_1, weightTensor_2, weightTensor_3, weightTensor_4,
                layer1_outputTensor, layer2_outputTensor, layer3_outputTensor, layer4_outputTensor);

            UpdateWeights(dataTensor[i], layer1_outputTensor, layer2_outputTensor, layer3_outputTensor, layer4_outputTensor,
                labelTensor[i], weightTensor_1, weightTensor_2, weightTensor_3, weightTensor_4, learningRate);
        }
        double batchLoss = BatchLoss(predictedTensor, labelTensor, batchSize);
        if (batchLoss < lowestLoss * 0.99) {
            lowestLoss = batchLoss;
            lowestLossEpoch = epoch;
        }
        if (epoch - lowestLossEpoch > patience && epoch > initialEpoch / 2) {
            printf("Early stopping at epoch %d   BatchLoss = %f\n", epoch, batchLoss);
            break;
        }
        if (epoch < 100 || epoch % 100 == 0) {
            printf("Epoch %d  BatchLoss = %f  lr = %f\n", epoch, batchLoss, learningRate);
        }
    }

    // ����ѭ�� 1 ��ε�ʱ��
    double x[2] = { 0, 0 };
    double prediction;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        x[0] = (rand() % 1000 / 1000.0);
        x[1] = (rand() % 1000 / 1000.0);
        prediction = Forward(x, weightTensor_1, weightTensor_2, weightTensor_3, weightTensor_4,
            layer1_outputTensor, layer2_outputTensor, layer3_outputTensor, layer4_outputTensor);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        x[0] = (rand() % 1000 / 1000.0);
        x[1] = (rand() % 1000 / 1000.0);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1 - (t3 - t2));
    printf("Time elapsed: %f nanoseconds\n", duration.count() / 10000.0);

    // 0.35 0.35 �� 0.25 0.55 ����Բ�⣬0.35 0.25 �� 0.15 0.45 ����Բ��
    x[0] = 0.35;
    x[1] = 0.35;
    prediction = Forward(x, weightTensor_1, weightTensor_2, weightTensor_3, weightTensor_4,
        layer1_outputTensor, layer2_outputTensor, layer3_outputTensor, layer4_outputTensor);
    printf("%f^2/0.16 + %f^2/0.36 = %f  prediction = %f���ж�С��1�ĸ��ʣ�\n", x[0], x[1], x[0] * x[0] / 0.16 + x[1] * x[1] / 0.36, prediction);

    return;
}