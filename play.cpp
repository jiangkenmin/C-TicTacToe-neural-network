#include <stdio.h>

#include "2_layers_dense_nn.h"

// tic tac toe game
static void GenerateMove(int board[9], int* avaiableMoves, int& numOfMoves) {
	for (int i = 0; i < 9; i++) {
		if (board[i] == 0) {
			avaiableMoves[numOfMoves++] = i;
		}
	}
}
static void PrintBoard(int board[9], int step) {
	printf("\n");
	printf("Step: %d\n", step);
	for (int i = 0; i < 9; i++) {
		if (board[i] == 1) printf("1 ");
		else if (board[i] == 2) printf("2 ");
		else printf("0 ");
		if (i % 3 == 2) printf("\n");
	}
	printf("\n");
}

// 1 为电脑先手，2 为电脑后手
void Play(int computer) {
	Load_2layers_NN_Weight();
	//Print_2Layers_NN_Weight();
	printf("Welcom to the game!\n\n");
	int board[9] = { 0 };
	int turn = 1;
	int step = 0;
	while (step <= 9) {
		PrintBoard(board, step);
		if (turn == computer) {
			int avaiableMoves[9] = { 0 };
			int numOfMoves = 0;
			GenerateMove(board, avaiableMoves, numOfMoves);
			printf("available moves: "); for (int i = 0; i < numOfMoves; i++) printf("%d ", avaiableMoves[i]); printf("\n");
			double maxScore = -10.0;
			int bestMove = -1;
			for (int i = 0; i < numOfMoves; i++) {
				board[avaiableMoves[i]] = turn; turn = 3 - turn;
				// begin evaluate
				double boardTensor[19] = { 0.0 };
				for (int j = 0; j < 9; j++) {
					if (board[j] == 1)
						boardTensor[j * 2 + 0] = 1.0;
					else if (board[j] == 2)
						boardTensor[j * 2 + 1] = 1.0;
				}
				if (turn == 1)
					boardTensor[18] = 0.0;
				else
					boardTensor[18] = 1.0;
				double score = -Forward_2Layers_NN(boardTensor);// PrintBoard(board, step);printf("move = %d, score = %f\n", avaiableMoves[i], score); for (int n = 0; n < 19; n++) printf("%f ", boardTensor[n]); printf("\n");
				// evaluate ended
				board[avaiableMoves[i]] = 0; turn = 3 - turn;

				if (score > maxScore) {
					maxScore = score;
					bestMove = avaiableMoves[i];
				}
			}
			if (bestMove == -1) printf("No valid move!\n");
			else {
				board[bestMove] = turn;
				printf("Computer move: %d, score: %f\n", bestMove + 1, maxScore);
			}
		}
		else {
			printf("Please enter your move (1-9): ");
			int move;
			scanf_s("%d", &move); printf("\n");
			move--;
			if (move >= 0 && move <= 8) {
				if (board[move] != 0) {
					printf("Invalid move!\n");
					continue;
				}
				board[move] = turn;
			}
			else {
				printf("Invalid move!\n");
				continue;
			}
		}
		turn = 3 - turn;
		step++;
	}
	return;
}