#pragma once
#include "constant.h"
#include <cstring>
#include <iostream>
#define X2BIT(x, M) (1 << (M - x - 1))
class Phase {
public:
    unsigned char top[MAX_N]; // 当前棋局每一列最顶部的空位置
    unsigned short user[MAX_N], machine[MAX_N]; // 用 16bits 记录一列中对方 && 己方的棋子位置
    int moves; // 记录当前步数
    static int M, N;    // 棋盘 x，y 方向上的高度和宽度
    static int noX, noY;    // 非落子点
    Phase() //生成一个空棋盘
        : moves(0)
    {
        memset(user, 0, sizeof(user));
        memset(machine, 0, sizeof(machine));
        memset(top, 0, sizeof(top));
    }

    Phase(int** board, const int*&& _top)   //根据给定状态 board 生成棋盘
        : moves(0)
    {
        memset(user, 0, sizeof(user));
        memset(machine, 0, sizeof(machine));
        memset(top, 0, sizeof(top));
        for (int i = 0; i < N; ++i)
            top[i] = _top[i];
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                setPiece(i, j, board[i][j]);
                if (board[i][j])
                    ++moves;
            }
        }
    }

private:
    bool alignment(const unsigned int* pos) const {
        // 水平方向
        unsigned int m[MAX_N] = {0};
        for (int i = 0; i < N - 1; ++i) m[i] = pos[i] & pos[i + 1];
        for (int i = 0; i < N - 3; ++i)
            if (m[i] & m[i + 2]) return true;

        // 垂直方向
        for (int i = 0; i < N; ++i) {
            m[i] = pos[i] & (pos[i] >> 1);
            if (m[i] & (m[i] >> 2)) return true;
        }

        // 左下到右上
        for (int i = 0; i < N - 1; ++i) m[i] = pos[i] & (pos[i + 1] >> 1);
        for (int i = 0; i < N - 3; ++i)
            if (m[i] & (m[i + 2] >> 2)) return true;

        // 右上到左下
        for (int i = 0; i < N - 1; ++i) m[i] = pos[i] & (pos[i + 1] << 1);
        for (int i = 0; i < N - 3; ++i)
            if (m[i] & (m[i + 2] << 2)) return true;

        return false;
    }

    void setPiece(int x, int y, int player)
    {
        if (player == 1) { // player==1 玩家
            user[y] |= X2BIT(x, M);
        } else if (player == 2) { // player==2 对方
            machine[y] |= X2BIT(x, M);
        } else {
            return;
        }
    }
};
