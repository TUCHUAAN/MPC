#include "OsqpEigen/OsqpEigen.h"
#include <iostream>
#include <Eigen/Dense>

class MPC
{
public:
	MPC(int n, int r, int m, int N, int a);




	void Update();//更新二次规划参数
	bool Solve_OsqpEigen();


private:

	//用户应设定
	Eigen::MatrixXd A;//状态矩阵 n*n
	Eigen::MatrixXd B;//输入矩阵 n*r
	Eigen::MatrixXd C;//输出矩阵 m*n

	Eigen::MatrixXd Q;//误差系数
	Eigen::MatrixXd F;//终端误差系数
	Eigen::MatrixXd R;//输入系数


	Eigen::VectorXd x;//状态变量 应设置初始值 默认为全0

	Eigen::SparseMatrix<double> A_q; //A: a*r矩阵,必须为稀疏矩阵SparseMatrix //输入约束矩阵
	Eigen::VectorXd lowerBound_q;                  //L: m*1下限向量 //输入约束
	Eigen::VectorXd upperBound_q;                  //U: m*1上限向量 //输入约束

	Eigen::VectorXd y;//输出向量 m*1
	Eigen::MatrixXd ref;//输出参考 m*N

	int n;  // 状态变量的维数
	int r;  // 输入的维数
	int m;  // 输出的维数
	int N; // 预测步长
	int a; //Ax约束数


	
	//
	Eigen::MatrixXd G;//二次规划求解 xT G x+g0x 最小值 代价函数
	Eigen::MatrixXd g0;//二次规划求解 xT G x+g0x 最小值 代价函数
	//求解器用

	Eigen::VectorXd QPSolution;//二次规划求解结果
    Eigen::MatrixXd Cp;//中间矩阵变量
	Eigen::MatrixXd Q1;//中间矩阵变量
	Eigen::MatrixXd Q2;//中间矩阵变量
	Eigen::MatrixXd M;//中间矩阵变量
	Eigen::MatrixXd R1;//中间矩阵变量

	//输出
	Eigen::VectorXd u;//输出向量 r*1
};



