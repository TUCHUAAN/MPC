#pragma once

#include "OsqpEigen/OsqpEigen.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class MPC
{
public:
    /**
     * @brief 构造函数
     * @param n 状态维数
     * @param r 输入维数
     * @param m 输出维数
     * @param N 预测步长
     * @param a 单步输入约束数（A_q 的行数）
     */
    MPC(int n, int r, int m, int N, int a);

    /// 更新二次规划参数（根据当前 x、ref 等重新构造 G、g0）
    void Update();

    /// 使用 OsqpEigen 求解 QP，得到当前控制量 u
    bool Solve_OsqpEigen();

    // ========================= 外部设置接口 =========================
    /// 设置系统矩阵 A, B, C
    void setSystemMatrices(const Eigen::MatrixXd& A,
        const Eigen::MatrixXd& B,
        const Eigen::MatrixXd& C);

    /// 设置权重矩阵 Q, R, F（其中 Q/F 是输出误差权重，维度 m×m，R 是输入权重 r×r）
    void setWeightMatrices(const Eigen::VectorXd& Q_vec,
        const Eigen::VectorXd& R_vec,
        const Eigen::VectorXd& F_vec);

    /// 设置当前状态 x（维度 n×1）
    void setState(const Eigen::VectorXd& x);

    /// 设置预测参考轨迹 ref，维度 m×(N+1)，每一列是一步的输出期望
    void setReference(const Eigen::MatrixXd& ref);

    /**
     * @brief 设置输入约束 A_q * u_k ∈ [lower, upper]
     *        输入为稠密矩阵 / 向量，类内部转换为稀疏矩阵
     * @param A_dense 约束矩阵 a×r
     * @param lower    下界向量 a×1
     * @param upper    上界向量 a×1
     */
    void setInputConstraints(const Eigen::MatrixXd& A_dense,
        const Eigen::VectorXd& lower,
        const Eigen::VectorXd& upper);

    /// 获取当前优化得到的控制量 u（r×1）
    const Eigen::VectorXd& getControl() const { return this->u; }

private:
    // ========================= 用户应设定 =========================
    Eigen::MatrixXd A;   // 状态矩阵 n×n
    Eigen::MatrixXd B;   // 输入矩阵 n×r
    Eigen::MatrixXd C;   // 输出矩阵 m×n

    Eigen::MatrixXd Q;   // 输出误差权重 m×m
    Eigen::MatrixXd F;   // 终端输出误差权重 m×m
    Eigen::MatrixXd R;   // 输入权重 r×r

    Eigen::VectorXd x;   // 当前状态 n×1

    Eigen::SparseMatrix<double> A_q; // 单步输入约束矩阵 a×r（稀疏）
    Eigen::VectorXd lowerBound_q;    // 单步约束下界 a×1
    Eigen::VectorXd upperBound_q;    // 单步约束上界 a×1

    Eigen::VectorXd y;   // 输出向量 m×1（可选用）
    Eigen::MatrixXd ref; // 输出参考轨迹 m×(N+1)

    int n;  // 状态维数
    int r;  // 输入维数
    int m;  // 输出维数
    int N;  // 预测步长
    int a;  // 单步约束数（A_q 行数）

    // ========================= QP 相关矩阵 =========================
    Eigen::MatrixXd G;   // 目标函数二次项矩阵（维度 N*r × N*r）
    Eigen::VectorXd g0;  // 目标函数一次项向量（维度 N*r × 1）

    Eigen::VectorXd QPSolution;  // QP 的解（维度 N*r × 1）

    // 中间矩阵变量
    Eigen::MatrixXd Cp;  // 预测控制对未来状态的影响矩阵 ((N+1)*n × N*r)
    Eigen::MatrixXd Q1;  // 堆叠的状态权重 ((N+1)*n × (N+1)*n)
    Eigen::MatrixXd Q2;  // 与参考轨迹相关的线性项 (1 × (N+1)*n)
    Eigen::MatrixXd M;   // 初始状态 x 对未来状态的影响 ((N+1)*n × n)
    Eigen::MatrixXd R1;  // 堆叠后的输入权重 (N*r × N*r)

    // 输出
    Eigen::VectorXd u;   // 当前时刻的输出控制量 r×1
};
