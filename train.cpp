#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nn_function.h"
#include "2_layers_dense_nn.h"

#include "tic_tac_toe_eng.h"


# define inputLength 19
# define batchNum 120000

static double dataTensor[batchNum][inputLength] = { 0 }; // 数据集
static double labelTensor[batchNum][1] = { 0 }; // 标签集

static double predictedTensor[batchNum][1] = { 0 }; // 保存训练时预测的结果，用于更新权重

void Train() {
    srand(18);
    Randomized_2Layers_NN_Weight(45);
    //Load_3layers_NN_Weight();

    //Print_2Layers_NN_Weight();
    // 生成 井字棋 数据集
    for (int n = 0; n < batchNum; n++) {
        int Rnd = rand() % 512;
        int step = 0;
        for (int i = 0; i <= 9; i++) {
            if (Rnd < (1 << i)) {
                step = i;
                break;
            }
        }
        int board[9] = { 0 };
        int player = 1;
        for (int i = 0; i < step; ) {
            int pos = rand() % 9;
            if (board[pos] == 0) {
                board[pos] = player;
                dataTensor[n][pos * 2 + player - 1] = 1.0;
                player = 3 - player;
                // 轮到玩家 1 则标记位为 0 ，玩家 2 则标记位为 1
                if (player == 1) dataTensor[n][inputLength - 1] = 0.0;
                else if (player == 2) dataTensor[n][inputLength - 1] = 1.0;
                if (i >= 4) {
                    if (check_winner(board, player))
                        break;
                }
                i++;
            }
        }
        // 引擎打分
        labelTensor[n][0] = gen_label(board, player);
        if (n < 10) {
            double prediction = Forward_2Layers_NN(dataTensor[n]);
            printf("Board: %d %d %d\n", board[0], board[1], board[2]);
            printf("Board: %d %d %d\n", board[3], board[4], board[5]);
            printf("Board: %d %d %d\n", board[6], board[7], board[8]);
            printf("Rnd: %d  Step: %d  Player: %d  Label: %f  Prediction: %f\n", Rnd, step, player, labelTensor[n][0], prediction);
        }
    }
    printf("Data generated.\n");

    // 可自定义参数
    const int batchSize = batchNum;
    const double initialLearningRate = 0.01;
    const int initialEpoch = 1000;
    const int patience = 100;
    const double learningRateDecayTo = 0.1; // 学习率衰减到原来的 learningRateDecayTo 倍

    double lowestLoss = 100000000.0;
    int lowestLossEpoch = 0;
    for (int epoch = 1; epoch <= initialEpoch; epoch++) {
        double learningRate = LearningRateDecay(epoch, initialLearningRate, initialEpoch, learningRateDecayTo);
        for (int i = 0; i < batchSize; i++) {
            predictedTensor[i][0] = Forward_2Layers_NN(dataTensor[i]);
            // printf("Loss = %f\n", Loss(predictedTensor[i][0], labelTensor[i][0]));
            UpdateWeights_2Layers_NN(dataTensor[i], labelTensor[i], learningRate);
        }
        double batchLoss = BatchLoss(predictedTensor, labelTensor, batchSize);
        if (batchLoss < lowestLoss * 0.998) {
            lowestLoss = batchLoss;
            lowestLossEpoch = epoch;
        }
        if (epoch - lowestLossEpoch > patience && epoch > initialEpoch / 2) {
            printf("Early stopping at epoch %d   BatchLoss = %.8f\n", epoch, batchLoss);
            break;
        }
        if (epoch < 100 || epoch % 1 == 0) {
            printf("Epoch %d/%d  BatchLoss = %.8f  lr = %f\n", epoch, initialEpoch, batchLoss, learningRate);
        }
    }
    Print_2Layers_NN_Weight();

    // 测试循环 1 万次的时间
    {
        double x[inputLength] = { 0 };
        double prediction;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; i++) {
            prediction = Forward_2Layers_NN(x);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; i++) {
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1 - (t3 - t2));
        printf("Time elapsed: %f nanoseconds\n", duration.count() / 10000.0);
    }

    // 预测
    printf("\n输入达到测试棋盘的步数\n");
    while (true) {
        double testDataTensor[1][inputLength] = { 0 };

        int step = 0;
        scanf_s("%d", &step); printf("\n");

        int board[9] = { 0 };
        int player = 1;
        for (int i = 0; i < step; ) {
            int pos = rand() % 9;
            if (board[pos] == 0) {
                board[pos] = player;
                testDataTensor[0][pos * 2 + player - 1] = 1;
                player = 3 - player;
                // 轮到玩家 1 则标记位为 0 ，玩家 2 则标记位为 1
                if (player == 1) testDataTensor[0][18] = 0;
                else if (player == 2) testDataTensor[0][18] = 1;
                if (i >= 4) {
                    if (check_winner(board, player))
                        break;
                }
                i++;
            }
        }
        // 引擎打分
        labelTensor[0][0] = gen_label(board, player);

        printf("Board: %d %d %d\n", board[0], board[1], board[2]);
        printf("Board: %d %d %d\n", board[3], board[4], board[5]);
        printf("Board: %d %d %d\n", board[6], board[7], board[8]);
        printf("Step: %d  Player: %d  Label: %f  ", step, player, labelTensor[0][0]);
        double prediction = Forward_2Layers_NN(testDataTensor[0]);
        printf("Prediction: %f\n\n", prediction);
    }
    return;
}