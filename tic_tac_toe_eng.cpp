
// tic tac toe engine

int check_winner(int board[9], int player) {
    // player 1 �� 2 �ֱ���� X �� O
    int win_conditions[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // ����
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // ����
        {0, 4, 8}, {2, 4, 6}            // �Խ���
    };

    for (int i = 0; i < 8; i++) {
        if (board[win_conditions[i][0]] != 0 &&
            board[win_conditions[i][0]] == board[win_conditions[i][1]] &&
            board[win_conditions[i][1]] == board[win_conditions[i][2]]) {
            if (board[win_conditions[i][0]] == player) {
                return 100; // ��һ�ʤ
            }
            else {
                return -100; // �������
            }
        }
    }
    return 0; // ƽ��
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
        int curscore = check_winner(board, player); // �ȼ���Ƿ���ʤ��
        if (curscore != 0) 
            return curscore; // ʤ��ֱ�ӷ��ط���
    }
    int bestscore = -100;
    for (int i = 0; i < 9; i++) {
        if (board[i] != 0) 
            continue; // �Ѿ������ӵ�λ�ò��ٿ���
        board[i] = player; // �����ڸ�λ������
        int score = -minimax(board, 3 - player); // �ݹ�����
        board[i] = 0; // ������λ�õ�����
        if (score > bestscore) {
            bestscore = score;
        }
    }
    return bestscore; // ��ѷ���
}

double gen_label(int board[9], int player) {
    int score = minimax(board, player); // ���㵱ǰ��ֵ���ѷ���
    if (score == 100) // win
        return 1.0;
    else if (score == 0) // draw
        return 0.0;
    else if (score == -100) // lose
        return -1.0;
}