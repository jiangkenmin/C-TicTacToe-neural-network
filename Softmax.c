static void Softmax(double* inputTensor, double outputTensor[], int inputSize) {
    double maxVal = 0;
    double sum = 0;
    // 找到输入数组中的最大值，防止指数函数溢出
    for (int i = 0; i < inputSize; i++) {
        if (inputTensor[i] > maxVal) {
            maxVal = inputTensor[i];
        }
    }
    // 计算指数和
    for (int i = 0; i < inputSize; i++) {
        outputTensor[i] = exp(inputTensor[i] - maxVal);
        sum += outputTensor[i];
    }

    for (int i = 0; i < inputSize; i++) {
        outputTensor[i] /= sum; // outputTensor 是 Softmax 层的输出
    }
}

static void SoftmaxBackward(double* inputTensor, double* outputTensor, double* gradOutput, double* gradInput, int inputSize) {
    // inputTensor 是 Softmax 层的输入。outputTensor 是 Softmax 层的输出
    // gradOutput 是损失函数对每个输出的偏导数。gradInput 是 Softmax 层的梯度，这是我们需要计算的梯度。
    for (int i = 0; i < inputSize; i++) {
        gradInput[i] = 0;
        for (int j = 0; j < inputSize; j++) {
            if (i == j) {
                gradInput[i] += outputTensor[i] * (1 - outputTensor[i]) * gradOutput[j];
            }
            else {
                gradInput[i] -= outputTensor[i] * outputTensor[j] * gradOutput[j];
            }
        }
    }
}


#include <stdio.h>

int main() {
    double inputTensor[] = {2.0, 1.0, 0.1};
    double outputTensor[3];
    double gradOutput[] = {0.1, -0.2, 0.1}; // 假设的损失函数对输出的梯度
    double gradInput[3];
    int inputSize = 3;

    // 计算 softmax 输出
    Softmax(inputTensor, outputTensor, inputSize);

    // 计算 softmax 反向传播的梯度
    SoftmaxBackward(inputTensor, outputTensor, gradOutput, gradInput, inputSize);

    for (int i = 0; i < inputSize; i++) {
        printf("outputTensor[%d] = %f\n", i, outputTensor[i]);
        printf("gradInput[%d] = %f\n", i, gradInput[i]);
    }

    return 0;
}
