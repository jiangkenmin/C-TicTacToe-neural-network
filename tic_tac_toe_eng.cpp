
// tic tac toe engine

int check_winner(int board[9], int player) {
    // player 1 和 2 分别代表 X 和 O
    int win_conditions[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // 横向
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // 纵向
        {0, 4, 8}, {2, 4, 6}            // 对角线
    };

    for (int i = 0; i < 8; i++) {
        if (board[win_conditions[i][0]] != 0 &&
            board[win_conditions[i][0]] == board[win_conditions[i][1]] &&
            board[win_conditions[i][1]] == board[win_conditions[i][2]]) {
            if (board[win_conditions[i][0]] == player) {
                return 100; // 玩家获胜
            }
            else {
                return -100; // 玩家输了
            }
        }
    }
    return 0; // 平局
}
static bool is_full(int board[9]) {
    for (int i = 0; i < 9; i++) {
        if (board[i] == 0) {
            return false;
        }
    }
    return true;
}

static int minimax(int board[9], int player) {
    if (is_full(board)) {
        return check_winner(board, player);
    }
    else {
        int curscore = check_winner(board, player); // 先检查是否有胜者
        if (curscore != 0) 
            return curscore; // 胜者直接返回分数
    }
    int bestscore = -100;
    for (int i = 0; i < 9; i++) {
        if (board[i] != 0) 
            continue; // 已经有棋子的位置不再考虑
        board[i] = player; // 尝试在该位置落子
        int score = -minimax(board, 3 - player); // 递归搜索
        board[i] = 0; // 撤销该位置的落子
        if (score > bestscore) {
            bestscore = score;
        }
    }
    return bestscore; // 最佳分数
}

double gen_label(int board[9], int player) {
    int score = minimax(board, player); // 计算当前棋局的最佳分数
    if (score == 100) // win
        return 1.0;
    else if (score == 0) // draw
        return 0.0;
    else if (score == -100) // lose
        return -1.0;
}