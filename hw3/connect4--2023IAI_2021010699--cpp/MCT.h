#pragma once
#include "Judge.h"
#include "Point.h"
#include "constant.h"
#include <assert.h>
#include <cstring>
#include <iostream>
#include <time.h>
#include <vector>

#define NextPlayer(player) (3 - player)
using namespace std;

struct Node {
    int cnt;
    double value;
    int player;
    int parent;
    int child[MAX_N];
    bool isLeaf()
    {
        for (int i = 0; i < MAX_N; i++)
            if (child[i] != -1)
                return false;
        return true;
    }
    double score()
    {
        if (cnt == 0)
            return 0;
        return value / cnt;
    }
};

struct NodePool {
    int newNode(int player, int parent)
    {
        pool[size].player = player;
        pool[size].parent = parent;
        pool[size].cnt = pool[size].value = 0;
        memset(pool[size].child, -1, sizeof(pool[size].child));
        size++;
        return size - 1;
    }
    void printPool(int root)
    {
        int cnt = 0;
        cerr << "root:" << root << endl;
        cerr << "++++ index ++ player ++ parent ++ cnt ++ value +++++" << endl;
        cerr << root << " " << pool[root].player << " " << pool[root].parent << " " << pool[root].cnt << " " << pool[root].value << endl;
        for (int i = 0; i < MAX_N; i++) {
            if (pool[root].child[i] != -1) {
                int ix = pool[root].child[i];
                cerr << ix << " " << pool[ix].player << " " << pool[ix].parent << " " << pool[ix].cnt << " " << pool[ix].value << endl;
            }
        }
        cerr << "++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    }
    Node& operator[](int idx) { return pool[idx]; }
    Node pool[NODEPOOL_SIZE];
    int size;
};

struct Phase {
    int player;
    int** board;
    int noX, noY;
    int M, N;
    int top[MAX_N];
    int able_place_col_num; // 可以落子的列数量
    int weight_sum; // 所有可以落子的点的权重和
    int weight[MAX_N]; // 每个可以落子的点的权重
    int middle;
    Phase()
    {
        board = new int*[MAX_M];
        for (int i = 0; i < MAX_M; i++) {
            board[i] = new int[MAX_N];
        }
    }
    Phase(const int m, const int n, const int* _top, int** _board, const int noX, const int noY)
        : M(m)
        , N(n)
        , noX(noX)
        , noY(noY)
        , player(2)
    {
        board = new int*[MAX_M];
        for (int i = 0; i < MAX_M; i++) {
            board[i] = new int[MAX_N];
        }
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = _board[i][j];
            }
        }
        able_place_col_num = 0;
        for (int i = 0; i < N; i++) {
            top[i] = _top[i];
            if (top[i]) {
                able_place_col_num++;
            }
        }
    }
    void setTo(const Phase& other)
    {
        player = 2;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = other.board[i][j];
            }
        }
        for (int i = 0; i < N; i++) {
            top[i] = other.top[i];
        }
        able_place_col_num = other.able_place_col_num;
        weight_sum = other.weight_sum;
    }
    Phase(Phase& phase)
    {
        M = phase.M;
        N = phase.N;
        noX = phase.noX;
        noY = phase.noY;
        player = phase.player;
        board = new int*[MAX_M]; // board[i][j] = 0 means empty space.
        for (int i = 0; i < MAX_M; i++) {
            board[i] = new int[MAX_N];
        }
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = phase.board[i][j];
            }
        }
        able_place_col_num = 0;
        for (int i = 0; i < N; i++) {
            top[i] = phase.top[i];
            if (top[i]) {
                able_place_col_num++;
            }
        }
    }
    void nextPhase(int y)
    {
        board[top[y] - 1][y] = player;
        if (top[y] >= 2 && noX == top[y] - 2 && noY == y) {
            top[y] -= 2;
        } else {
            top[y]--;
        }
        if (!top[y]) {
            able_place_col_num--;
            weight_sum -= weight[y];
        }
        player = NextPlayer(player);
    }
    void printPhase()
    {
        cerr << "============= PrintPhase =============" << endl;
        cerr << "M: " << M << " N: " << N << " Player: " << player << endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (noX == i && noY == j) {
                    cerr << "x ";
                } else {
                    if (board[i][j] == 0) {
                        cerr << "  ";
                    } else if (board[i][j] == 1) {
                        cerr << "M ";
                    } else {
                        cerr << "E ";
                    }
                }
            }
            cerr << endl;
        }
        cerr << "============= END =============" << endl;
    }
    bool boardFull()
    {
        return !able_place_col_num;
    }
    int moveScore(int move, int player)
    {
        int score = 1; // the score of the player to move this move.
        score += align3(move, player);
        if (score > 1000)
            return score;
        score += align3(move, NextPlayer(player));
        if (score > 1000)
            return score;
        score += (move < (N >> 1)) ? move / 3 : (N - 1 - move) / 3;
        return score;
    }
    int align3(int move, int player)
    {
        int left = max(move - 3, 0);
        int right = min(move + 3, N - 1);
        int move_x = top[move] - 1;
        int score = 0;
        // int bef_x = top[move];
        // if (noX == top[move] - 1 && noY == move) {
        //     move_x = top[move] - 2;
        // } else {
        //     move_x = top[move] - 1;
        // }
        board[move_x][move] = player;
        int enemy_num = 0;
        int my_num = 0;
        // 水平方向
        for (int j = 0; j < 4; j++) {
            if (board[move_x][left + j] == player) {
                my_num++;
            } else if (board[move_x][left + j] == NextPlayer(player)) {
                enemy_num++;
            }
        }
        for (int i = left + 1; i <= right - 3; i++) {
            if (my_num == 4) {
                // 复位
                board[move_x][move] = 0;
                return 1000000;
            } else if (my_num == 3 && enemy_num == 0) {
                score += 1;
            }
            if (board[move_x][i] == player) {
                my_num--;
            } else if (board[move_x][i] == NextPlayer(player)) {
                enemy_num--;
            }
            if (board[move_x][i + 3] == player) {
                my_num++;
            } else if (board[move_x][i + 3] == NextPlayer(player)) {
                enemy_num++;
            }
        }
        // 对角线左上到右下
        enemy_num = 0;
        my_num = 0;
        for (int j = 0; j < 4; j++) {
            if (board[move_x - 3 + j][left + j] == player) {
                my_num++;
            } else if (board[move_x - 3 + j][left + j] == NextPlayer(player)) {
                enemy_num++;
            }
        }
        for (int i = left + 1; i <= right - 3; i++) {
            if (my_num == 4) {
                // 复位
                board[move_x][move] = 0;
                return 1000000;
            } else if (my_num == 3 && enemy_num == 0) {
                score += 1;
            }
            if (board[move_x - (move - i)][i] == player) {
                my_num--;
            } else if (board[move_x - (move - i)][i] == NextPlayer(player)) {
                enemy_num--;
            }
            if (board[move_x - (move - i) + 3][i + 3] == player) {
                my_num++;
            } else if (board[move_x - (move - i) + 3][i + 3] == NextPlayer(player)) {
                enemy_num++;
            }
        }

        // 对角线右上到左下
        enemy_num = 0;
        my_num = 0;
        for (int j = 0; j < 4; j++) {
            if (board[move_x + 3 - j][left + j] == player) {
                my_num++;
            } else if (board[move_x + 3 - j][left + j] == NextPlayer(player)) {
                enemy_num++;
            }
        }
        for (int i = left + 1; i <= right - 3; i++) {
            if (my_num == 4) {
                // 复位
                board[move_x][move] = 0;
                return 1000000;
            } else if (my_num == 3 && enemy_num == 0) {
                score += 1;
            }
            if (board[move_x + (move - i)][i] == player) {
                my_num--;
            } else if (board[move_x + (move - i)][i] == NextPlayer(player)) {
                enemy_num--;
            }
            if (board[move_x + (move - i) - 3][i + 3] == player) {
                my_num++;
            } else if (board[move_x + (move - i) - 3][i + 3] == NextPlayer(player)) {
                enemy_num++;
            }
        }

        // 竖直方向上
        enemy_num = 0;
        my_num = 0;
        if (move_x < M - 2) {
            for (int i = 0; i < 3; i++) {
                if (board[move_x + i][move] == player) {
                    my_num++;
                }
            }
        }
        if (my_num == 3) {
            score += 1;
            if (move_x < M - 3) {
                if (board[move_x + 3][move] == player) {
                    // 复位
                    board[move_x][move] = 0;
                    return 1000000;
                }
            }
        }
        // 复位
        board[move_x][move] = 0;
        return score; // 未找到胜利者的概率估计值应该是游戏的开始力
    }
    bool isWinningMove(int y, int player);
};

class MCTree {
public:
    MCTree()
        : init_phase()
        , cur_phase()
        , lastMove(-1)
    {
    }

    int search(int timeLimit, int lastY)
    {
        int now_lastY = lastY; //当前所处结点的前一步行棋坐标
        int init_root = root;
        time_t start = clock();
        int search_count = 0;
        while (true) {
            // if(search_count>nodes[root].cnt){
            //     cerr<<"search_count>nodes[root].cnt "<<endl;
            //     cerr<<"search_count: "<<search_count<<endl;
            //     nodes.printPool(root);
            // }
            search_count++;
            int usedTime = clock() - start;
            if (usedTime >= timeLimit) {
                break;
            }
            now_lastY = lastY;  //重置上一步行棋子位置
            cur_phase.setTo(init_phase);
            int cur = select(now_lastY);
            // if(cur == root){
            //     cerr<<"Invilid select"<<endl;
            // }
            // if(cur == -1){
            //     cerr<<"cur == -1,select error #1";
            //     // cerr<<"cur: "<<cur<<endl;
            //     // for(int i=0;i<cur_phase.N;i++){
            //     //     cerr<<"child["<<i<<"]: "<<nodes[cur].child[i]<<endl;
            //     // }
            // }
            int ans_cur = cur;
            if (nodes[cur].cnt) {
                cur = expand(cur, init_root, now_lastY);
            }
            // if(cur == -1){
            //     cerr<<"ans_cur: "<<ans_cur<<endl;
            //     cerr<<"pool_size: "<<nodes.size<<endl;
            //     nodes.printPool(ans_cur);
            //     cerr<<"cur == -1,select error #2";
            // }
            backUp(cur, rollout(cur, now_lastY));
        }
        // nodes.printPool(root);
        cerr << "search_count: " << search_count << endl;
        cerr << "pool_size: " << nodes.size << endl;
        return lastMove = finalDecision();
    }
    void setPhase(const int M, const int N, const int* top, int** _board, const int noX, const int noY, const int lastY)
    {
        // 初始化 init_phase, cur_phase
        init_phase.player = cur_phase.player = 2;
        init_phase.M = cur_phase.M = M;
        init_phase.N = cur_phase.N = N;
        init_phase.noX = cur_phase.noX = noX;
        init_phase.noY = cur_phase.noY = noY;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                init_phase.board[i][j] = cur_phase.board[i][j] = _board[i][j];
            }
        }
        cur_phase.weight_sum = init_phase.weight_sum = 0;
        int middle = N >> 1;
        init_phase.middle = cur_phase.middle = middle;
        init_phase.able_place_col_num = 0;
        for (int i = 0; i < N; i++) {
            init_phase.top[i] = cur_phase.top[i] = top[i];
            if (top[i]) {
                init_phase.able_place_col_num++;
                int weight = middle > i ? i + 1 : N - i;
                cur_phase.weight[i] = init_phase.weight[i] = weight;
                init_phase.weight_sum += weight;
            }
        }
        cur_phase.able_place_col_num = init_phase.able_place_col_num;
        cur_phase.weight_sum = init_phase.weight_sum;

        if (lastMove == -1 || (NODEPOOL_SIZE - nodes.size) <= 3e6) {
            nodes.size = 0;
            root = nodes.newNode(2, -1);
        } else {
            moveRoot(lastMove);
            moveRoot(lastY);
            // nodes.printPool(root);
        }
        if (nodes[root].isLeaf()) {
            expand(root);
        }
        // cerr<<"set end"<<endl;
        // for(int i = 0;i<N;i++){
        //     cerr<< init_phase.weight[i]<<" ";
        // }
        // cerr<<endl;
        // for(int i = 0;i<N;i++){
        //     cerr<< cur_phase.weight[i]<<" ";
        // }
        // cerr<<endl;
    }

private:
    Phase init_phase, cur_phase;
    NodePool nodes;
    int root;
    int lastMove;
    int nextMove[MAX_N];
    int score[MAX_N];
    int moveValue[MAX_M][MAX_N];
    int moveCnt[MAX_M][MAX_N];

    void moveRoot(int move);
    int bestMove(int node);
    void backUp(int node, int value);
    int rollout(int node, int lastY = -1);
    int expand(int node, int init_root = 0, int lastY = -1);
    int select(int &now_lastY);
    int finalDecision();
    int smartTry(int player);
    int scoreSample(int valid_num, int player);
    // int centerTry(int player, Phase rollout_phase);
};