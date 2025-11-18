#include"MPC.h"



MPC::MPC(int n, int r, int m, int N, int a)
{
	this->n = n;
	this->r = r;
	this->m = m;
	this->N = N;
	this->a = a;
	this->A = Eigen::MatrixXd::Zero(n, n);
	this->B = Eigen::MatrixXd::Zero(n, r);
	this->C = Eigen::MatrixXd::Zero(m, n);



	this->x = Eigen::VectorXd::Zero(n);//状态变量
	this->u = Eigen::VectorXd::Zero(r);//

	this->y = Eigen::VectorXd::Zero(m);
	this->ref = Eigen::VectorXd::Zero(m, (N + 1));


	this->G = Eigen::MatrixXd::Zero(N * r, N * r);
	this->g0 = Eigen::MatrixXd::Zero(1, N * r);

	this->Cp = Eigen::MatrixXd::Zero((N + 1) * n, N * r);
	this->M = Eigen::MatrixXd::Zero((N + 1) * n, n);
	this->Q1 = Eigen::MatrixXd::Zero((N + 1) * n, (N + 1) * n);
	this->Q2 = Eigen::MatrixXd::Zero(1, (N + 1) * n);
	this->R1 = Eigen::MatrixXd::Zero(N * r, N * r);

	//求解器2
	A_q = Eigen::SparseMatrix<double>(a, r);//A: m*n矩阵,必须为稀疏矩阵SparseMatrix //a个约束
	lowerBound_q = Eigen::VectorXd::Zero(a);                  //L: m*1下限向量
	upperBound_q = Eigen::VectorXd::Zero(a);              //U: m*1上限向量


}

void MPC::Update()//更新二次规划参数 
{

	Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(this->n, this->n);//A n次方
	for (int i = 0; i < N + 1; i++)
	{

		Eigen::MatrixXd temp_AB = temp * B;
		M.block(i * this->n, 0, this->n, this->n) = temp;//赋值M
		temp = temp * A;


		for (int j = 1; j < N - i + 1; j++)
		{
			//cout << temp_AB << endl << endl;
			Cp.block((j + i) * this->n, (j - 1) * this->r, this->n, this->r) = temp_AB;//赋值CP
			//cout << Cp << endl << endl;
		}


	}


	Eigen::MatrixXd temp_Q = Eigen::MatrixXd::Zero(this->n, this->n);
	temp_Q = C.transpose() * Q * C;
	Eigen::MatrixXd temp_F = Eigen::MatrixXd::Zero(this->n, this->n);
	temp_F = C.transpose() * F * C;

	for (int i = 0; i < N + 1; i++)//赋值Q1
	{


		if (i == N)
		{
			Q1.block(i * this->n, i * this->n, this->n, this->n) = temp_F;

		}
		else
		{
			Q1.block(i * this->n, i * this->n, this->n, this->n) = temp_Q;

		}
	}


	for (int i = 0; i < N; i++)//赋值R1
	{
		R1.block(i * this->r, i * this->r, this->r, this->r) = R;
	}

	G = 2 * (Cp.transpose() * Q1 * Cp + R1);//赋值G





	Eigen::MatrixXd temp_rQ = Eigen::MatrixXd::Zero(1, n);
	temp_rQ = -2 * Q * C;
	Eigen::MatrixXd temp_rF = Eigen::MatrixXd::Zero(1, n);
	temp_rF = -2 * F * C;
	for (int i = 0; i < N + 1; i++)//赋值Q2
	{


		if (i == N)
		{
			Q2.block(0, i * this->n, 1, this->n) = ref.block(0, i, this->m, 1) * temp_rF;

		}
		else
		{
			Q2.block(0, i * this->n, 1, this->n) = ref.block(0, i, this->m, 1) * temp_rQ;

		}
	}



	g0 = 2 * x.transpose() * M.transpose() * Q1 * Cp + Q2 * Cp;



}


bool MPC::Solve_OsqpEigen()
{

	// allocate QP problem matrices and vectores
	Eigen::SparseMatrix<double>hessian(N * r, N * r);
	Eigen::VectorXd gradient(N * r);                    //Q: n*1向量


	for (int i = 0; i < N * r; i++)
	{
		for (int j = 0; j < N * r; j++)
		{
			hessian.insert(i, j) = G(i, j);
		}
	}

	//std::cout << "hessian:" << std::endl
	//          << hessian << std::endl;
	gradient = g0.transpose();


	Eigen::SparseMatrix<double> linearMatrix(N * A_q.rows(), N * r); //A: m*n矩阵,必须为稀疏矩阵SparseMatrix
	Eigen::VectorXd lowerBound(N * A_q.rows());                  //L: m*1下限向量
	Eigen::VectorXd upperBound(N * A_q.rows());                  //U: m*1上限向量



	for (int k = 0; k < N; k++)//赋值linearMatrix
	{
		for (int i = 0; i < A_q.rows(); i++)
		{
			for (int j = 0; j < r; j++)
			{
				linearMatrix.insert(k * A_q.rows() + i, k * r + j) = A_q.coeff(i, j);
			}
		}
	}



	for (int i = 0; i < N; i++)
	{

		for (int j = 0; j < A_q.rows(); j++)
		{

			lowerBound[A_q.rows() * i + j] = lowerBound_q(j);

		}

	}

	for (int i = 0; i < N; i++)
	{

		for (int j = 0; j < A_q.rows(); j++)
		{

			upperBound[A_q.rows() * i + j] = upperBound_q(j);

		}

	}

	//std::cout << "hessian:" << std::endl
 //         << upperBound << std::endl;

	// instantiate the solver
	OsqpEigen::Solver solver;

	// settings
	solver.settings()->setVerbosity(false);
	solver.settings()->setWarmStart(true);
	//solver.settings()->setAbsoluteTolerance(1e-2); // 设置绝对收敛精度
	//solver.settings()->setRelativeTolerance(1e-2);  // 相对精度
	//solver.settings()->setMaxIteration(200); // 减少最大迭代次数
	//solver.settings()->setEpsAbs(1e-4);      // 放宽收敛精度

	// set the initial data of the QP solver
	solver.data()->setNumberOfVariables(N * r);   //变量数n
	solver.data()->setNumberOfConstraints(linearMatrix.rows()); //约束数a
	if (!solver.data()->setHessianMatrix(hessian))
		return false;
	if (!solver.data()->setGradient(gradient))
		return false;
	if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
		return false;
	if (!solver.data()->setLowerBound(lowerBound))
		return false;
	if (!solver.data()->setUpperBound(upperBound))
		return false;

	// instantiate the solver
	if (!solver.initSolver())
		return false;



	// solve the QP problem
	solver.solveProblem();

	QPSolution = solver.getSolution();
	for (int i = 0; i < r; i++)
	{
		u(i) = QPSolution(i);
	}
	//std::cout << "QPSolution" << std::endl
	//	<< QPSolution << std::endl; //输出为m*1的向量
	return true;


}



