#include "MCT.h"
#include <climits>
#include <cmath>
#include <random>
bool my_userWin(const int x, const int y, const int M, const int N, int** board)
{
    // 横向检测
    int i, j;
    int count = 0;
    for (i = y; i >= 0; i--)
        if (!(board[x][i] == 1))
            break;
    count += (y - i);
    for (i = y; i < N; i++)
        if (!(board[x][i] == 1))
            break;
    count += (i - y - 1);
    if (count >= 4)
        return true;

    // 纵向检测
    count = 0;
    for (i = x; i < M; i++)
        if (!(board[i][y] == 1))
            break;
    count += (i - x);
    if (count >= 4)
        return true;

    // 左下-右上
    count = 0;
    for (i = x, j = y; i < M && j >= 0; i++, j--)
        if (!(board[i][j] == 1))
            break;
    count += (y - j);
    for (i = x, j = y; i >= 0 && j < N; i--, j++)
        if (!(board[i][j] == 1))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    // 左上-右下
    count = 0;
    for (i = x, j = y; i >= 0 && j >= 0; i--, j--)
        if (!(board[i][j] == 1))
            break;
    count += (y - j);
    for (i = x, j = y; i < M && j < N; i++, j++)
        if (!(board[i][j] == 1))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    return false;
}

bool my_machineWin(const int x, const int y, const int M, const int N, int** board)
{
    // 横向检测
    int i, j;
    int count = 0;
    for (i = y; i >= 0; i--)
        if (!(board[x][i] == 2))
            break;
    count += (y - i);
    for (i = y; i < N; i++)
        if (!(board[x][i] == 2))
            break;
    count += (i - y - 1);
    if (count >= 4)
        return true;

    // 纵向检测
    count = 0;
    for (i = x; i < M; i++)
        if (!(board[i][y] == 2))
            break;
    count += (i - x);
    if (count >= 4)
        return true;

    // 左下-右上
    count = 0;
    for (i = x, j = y; i < M && j >= 0; i++, j--)
        if (!(board[i][j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i >= 0 && j < N; i--, j++)
        if (!(board[i][j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    // 左上-右下
    count = 0;
    for (i = x, j = y; i >= 0 && j >= 0; i--, j--)
        if (!(board[i][j] == 2))
            break;
    count += (y - j);
    for (i = x, j = y; i < M && j < N; i++, j++)
        if (!(board[i][j] == 2))
            break;
    count += (j - y - 1);
    if (count >= 4)
        return true;

    return false;
}

using namespace std;

int MCTree::bestMove(int node)
{ // 找到下一个最佳节点的位置并返回
    // nodes.printPool();
    int best = -1; // 最佳结点
    double maxScore = -INFINITY; // 记录最大得分
    double log_N = log((double)nodes[node].cnt + (double)(0.001));
    for (int i = 0; i < init_phase.N; ++i) {
        // cerr << "i:" << i << endl;
        int child = nodes[node].child[i];
        if (child == -1) {
            // cerr << "Invalid node, continue" << endl;
            continue;
        }
        if (!nodes[child].cnt) // 优先选择未被访问过的结点
        {
            return i;
        }
        //! UCB score
        double score = nodes[child].score() + (UCB_C) * sqrt(log_N / ((double)nodes[child].cnt + (double)0.001));
        //!  exploration bonus based on the prior
        // double score = nodes[child].score() + (cur_phase.weight[i] / (double)cur_phase.weight_sum) * (sqrt(1 + (double)nodes[node].cnt) / (1 + (double)nodes[child].cnt)) * (ALPHA1 + log((nodes[node].cnt + ALPHA2 + 1) / ALPHA2));
        
        
        
        if (score > maxScore) {
            maxScore = score;
            best = i;
        }
    }
    // cerr << "BestMove: " << best << endl;
    return best;
}
int MCTree::select(int& now_lastY)
{ // 寻找最佳节点并返回该节点的位置
    int node = root;
    int player = nodes[node].player;
    int ans_cnt = 0;
    while (!nodes[node].isLeaf()) {
        ans_cnt++;
        int move = bestMove(node);
        now_lastY = move;
        cur_phase.nextPhase(move);
        // if (nodes[nodes[node].child[move]].parent != node) {
        //     cerr << "Parent error!" << endl;
        //     cerr << "root: " << root << endl;
        //     cerr << "node: " << node << endl;
        //     cerr << "move: " << move << endl;
        //     cerr << "nodes[node].child[move]: " << nodes[node].child[move] << endl;
        //     nodes.printPool(node);
        // }
        node = nodes[node].child[move]; // 移动到该结点下面的一个子结点中
    }
    // if (!ans_cnt) {
    //     cerr << "select too short" << endl;
    //     nodes.printPool(root);
    // }
    return node;
}
int MCTree::expand(int node, int init_root, int lastY)
{ // 拓展结点
    int player = nodes[node].player;
    if (!cur_phase.able_place_col_num) { // 如果当前棋盘中棋子已满，则返回当前结点用于扩展
        return node;
    }
    if (lastY != -1) { // 如果当前棋局为某一方获胜，则返回该结点用于扩展
        int last_x = cur_phase.top[lastY] == cur_phase.noX && lastY == cur_phase.noY ? cur_phase.top[lastY] + 1 : cur_phase.top[lastY];
        if (my_userWin(last_x, lastY, cur_phase.M, cur_phase.N, cur_phase.board) || my_machineWin(last_x, lastY, cur_phase.M, cur_phase.N, cur_phase.board)) {
            return node;
        }
    }
    int exp_able_child_num = 0;
    int lose_move = -1;
    for (int i = 0; i < init_phase.N; i++) {
        if (cur_phase.top[i]) // 该点还可以落子
        {
            exp_able_child_num++;
            if (cur_phase.isWinningMove(i, player)) {
                for (int j = 0; j < init_phase.N; ++j)
                    nodes[node].child[j] = -1;
                nodes[node].child[i] = nodes.newNode(NextPlayer(player), node);
                cur_phase.nextPhase(i);
                // if (nodes[node].child[i] == -1) {
                //     cerr << "Invalid isWinningMove(i, player)" << endl;
                //     nodes.printPool(node);
                // }
                return nodes[node].child[i];
            }
            if (cur_phase.isWinningMove(i, NextPlayer(player))) {
                if(lose_move==-1){  //此前没有发现过对面必胜的位置，则由于该点为我方必输点，前面所有的孩子都没有意义了
                    for (int j = 0; j < i; ++j)
                        nodes[node].child[j] = -1;
                }
                nodes[node].child[i] = nodes.newNode(NextPlayer(player), node);
                lose_move = i;
                // cur_phase.nextPhase(i);
                // return nodes[node].child[i];
            }
            if(lose_move == -1){    //如果没有发现过对面必胜的位置，则可以继续探索可以开拓的点
                nodes[node].child[i] = nodes.newNode(NextPlayer(player), node);
            }
            // if (init_root && root == 0) {
            //     cerr << "root becomes 0 at here_10" << endl;
            //     cerr << "ans_size: " << ans_size << endl;
            //     cerr << "size: " << nodes.size << endl;
            //     nodes.printPool(root);
            //     nodes.printPool(init_root);
            // }
        }
    }
    if(lose_move >= 0){ //如果有必输点则优先拓展出必输点
        return nodes[node].child[lose_move];
    }
    int pos_y = rand() % exp_able_child_num;
    int ans_y = 0;
    for (int i = 0; i < init_phase.N; i++) {
        if (nodes[node].child[i] != -1) {
            if (ans_y == pos_y) {
                cur_phase.nextPhase(i);
                return nodes[node].child[i];
            }
            ans_y++;
        }
    }
    cerr << "error move" << endl;
    return node; // 结束节点的位置或节点编号。不要加上root节点的位置。或者加
}
int MCTree::rollout(int cur, int lastY)
{
    int cnt = 0;
    if (lastY != -1) {
        int last_x = cur_phase.top[lastY] == cur_phase.noX && lastY == cur_phase.noY ? cur_phase.top[lastY] + 1 : cur_phase.top[lastY];
        if (my_userWin(last_x, lastY, cur_phase.M, cur_phase.N, cur_phase.board)) {
            return (nodes[cur].player == 2); // win!! return 1 to play the move.
        }
        if (my_machineWin(last_x, lastY, cur_phase.M, cur_phase.N, cur_phase.board)) {
            return (nodes[cur].player == 1); // win!! return 1 to play the move.
        }
    }
    // int weight_before = cur_phase.weight_sum;
    while (true) {
        if (!cur_phase.able_place_col_num) {
            cerr << "board full" << endl;
            return rand() % 2;
        }
        cnt++;
        // int move = cur_phase.able_place_col[rand() % cur_phase.able_place_col_num];
        int move = -1;
        //! RANDOM POLICY BEGIN
        // int ans = rand() % cur_phase.able_place_col_num;
        // int ans_cnt = 0;
        // for (int i = 0; i < cur_phase.N; i++) {
        //     if (ans_cnt == ans) {
        //         move = i;
        //     }
        //     if (cur_phase.top[i]) {
        //         ans_cnt++;
        //     }
        // }
        //! RANDOM POLICY END
        // if(weight_before != cur_phase.weight_sum){

        //     cerr << " cur_phase.weight_sum: " << cur_phase.weight_sum << endl;
        //     bool flag = 0;
        //     for (int i = 0; i < cur_phase.N; i++) {
        //         if (cur_phase.top[i]) {
        //             cerr << cur_phase.weight[i] << " ";
        //         } else {
        //             cerr << "- ";
        //         }
        //     }
        //     cerr << endl;
        //     weight_before = cur_phase.weight_sum;
        // }
        //! CENTER POLICY BEGIN
        int ans = rand() % cur_phase.weight_sum;
        int ans_cnt = 0;
        for (int i = 0; i < cur_phase.N; i++) {
            if (cur_phase.top[i]) {
                ans_cnt += cur_phase.weight[i];
                if (ans_cnt > ans) {
                    move = i;
                    break;
                }
            }
        }
        //! CENTER POLICY END

        // cerr<<"cnt: "<<cnt<<" move: "<<move<<endl;
        // int move = rand() % cur_phase.N;
        // move = smartTry(cur_phase.player);
        // cerr<<"move: "<<move<<endl;
        // cur_phase.printPhase();

        cur_phase.nextPhase(move);
        // cur_phase.printPhase();
        int last_x = cur_phase.top[move] == cur_phase.noX && move == cur_phase.noY ? cur_phase.top[move] + 1 : cur_phase.top[move];
        if (my_userWin(last_x, move, cur_phase.M, cur_phase.N, cur_phase.board)) {
            // nodes[cur].cnt++;
            // nodes[cur].value = (nodes[cur].player == 1);
            // cur_phase.printPhase();
            return (nodes[cur].player == 2); // win!! return 1 to play the move.
        } else if (my_machineWin(last_x, move, cur_phase.M, cur_phase.N, cur_phase.board)) {
            // nodes[cur].cnt++;
            // nodes[cur].value = (nodes[cur].player == 2);
            // cur_phase.printPhase();
            return (nodes[cur].player == 1); // win!! return 1 to play the move.
        }
        // if (cnt > 100) {
        //     cerr << "count>100" << endl;
        //     return rand() % 2;
        // }
    }
    // cerr << "cnt: " << cnt << endl;
}
void MCTree::backUp(int cur, int rollout)
{
    // cerr << "backUp..." << endl;
    // if (cur == root) {
    //     // cerr << "cur==root! return" << endl;
    //     // cerr << "nodes[cur].cnt " << nodes[cur].cnt << " nodes[cur].value " << nodes[cur].value << endl;
    //     return;
    // }
    int cnt = 0;
    int cur_bef = root;
    while (cur != -1) {
        cur_bef = cur;
        nodes[cur].cnt++; // visits number.
        nodes[cur].value += rollout;
        cur = nodes[cur].parent;
        rollout = !rollout;
        cnt++;
        // if (cnt > 150) {
        //     cerr << "Too long back up" << endl;
        //     cerr << "cur: " << cur << endl;
        //     cerr << "nodes[cur].cnt: " << nodes[cur].cnt << " nodes[cur].value: " << nodes[cur].value << endl;
        //     cerr << "nodes[cur].parent: " << nodes[cur].parent << endl;
        // }
    }
    // if (cur_bef != root) {
    //     cerr << "cur==root! return" << endl;
    //     cerr << "cur_bef: " << cur_bef << endl;
    //     cerr << "root: " << root << endl;
    // }
    // if (!cnt) {
    //     cerr << "Too short back up" << endl;
    //     cerr << "cur_bef: " << cur_bef << endl;
    //     nodes.printPool(root);
    //     for (int i = 0; i < init_phase.N; i++) {
    //         cerr << nodes[root].child[i] << " ";
    //     }
    //     cerr << endl;
    // }
}
int MCTree::finalDecision()
{
    int minvisits = -1; //-1 means infinity.
    int bestmove = 0; // the move that wins the most simulations.
    int alert = -1;
    for (int i = 0; i < init_phase.N; i++) { // find the best move.
        // cerr<<"init_phase.toTop(init_phase.top[i], i): "<<init_phase.toTop(init_phase.top[i], i)<<" nodes[nodes[root].child[i]].cnt: "<<nodes[nodes[root].child[i]].cnt<<endl;
        if (init_phase.top[i]) {
            init_phase.board[init_phase.top[i] - 1][i] = 2;
            if (my_machineWin(init_phase.top[i] - 1, i, init_phase.M, init_phase.N, init_phase.board)) {
                return i; // win!! return 1 to play the move.
            }
            init_phase.board[init_phase.top[i] - 1][i] = 1;
            if (my_userWin(init_phase.top[i] - 1, i, init_phase.M, init_phase.N, init_phase.board)) {
                alert = i; // win!! return 1 to play the move.
            }
            init_phase.board[init_phase.top[i] - 1][i] = 0;
            if (init_phase.top[i] && nodes[nodes[root].child[i]].cnt > minvisits) { // if this move has been visited
                minvisits = nodes[nodes[root].child[i]].cnt; // it has been visited. So, it has the minimum number of visits.
                bestmove = i;
            }
        }
    }
    if (alert >= 0) {
        return alert;
    }
    // cerr << "best move: " << bestmove << endl;
    return bestmove; // return the move.
}
void MCTree::moveRoot(int move)
{
    if (nodes[root].child[move] == -1) {
        // cerr << "Move to invalid node ===> create new" << endl;
        root = nodes.newNode(NextPlayer(nodes[root].player), root);
    } else {
        // cerr << "child exist, move to it" << endl;
        root = nodes[root].child[move]; // move the root to the move.
    }
    nodes[root].parent = -1; // change the parent of the root to -1.
}
int MCTree::smartTry(int player)
{
    int valid_num = 0;
    for (int i = 0; i < cur_phase.N; i++) {
        if (cur_phase.top[i]) { // if this move is invalid, skip it.
            nextMove[valid_num++] = i;
            // cerr<<"i: "<<i<<endl;
        }
    }
    return nextMove[scoreSample(valid_num, player)];
}
int MCTree::scoreSample(int valid_num, int player)
{ // sample a number of valid moves. Return -1 means invalid move. Return
    int totalScore = 0;
    for (int i = 0; i < valid_num; ++i) {
        score[i] = cur_phase.moveScore(nextMove[i], player);
        // cerr<<"score["<<i<<"]: "<<score[i]<<" | "; // print the score.
        totalScore += score[i];
    }
    // cerr<<endl;
    if (totalScore == 0)
        return 0;
    // int randNum = getRandomNumber(totalScore + 1);
    int randNum = rand() % totalScore;
    // cerr<<"randNum:"<<randNum<<endl;
    int move = -1;
    do {
        ++move;
        randNum -= score[move];
    } while (randNum > 0);
    return move;
}

// int MCTree::centerTry(int player, Phase rollout_phase)
// {
//     int valid_num = 0;
//     int sgn = 0;
//     int center = rollout_phase.N >> 1;
//     for (int i = center; i >= 0 && i < rollout_phase.N; i = center + sgn) {
//         if (!rollout_phase.toTop(rollout_phase.top[i], i)) { // if this move is invalid, skip it.
//             return i;
//         }
//         sgn = -sgn; // change sign. This is to make the center move the best move.
//         if (sgn >= 0)
//             sgn++;
//     }
// }
bool Phase::isWinningMove(int y, int player)
{
    int nx = top[y] - 1;
    // if (noX == top[y] - 1 && noY == y) {
    //     nx = top[y] - 2;
    // } else {
    //     nx = top[y] - 1;
    // }
    board[nx][y] = player;
    if (player == 1) {
        if (my_userWin(nx, y, M, N, board)) {
            board[nx][y] = 0;
            return true;
        }
    } else {
        if (my_machineWin(nx, y, M, N, board)) {
            board[nx][y] = 0;
            return true;
        }
    }
    board[nx][y] = 0;
    return false;
}