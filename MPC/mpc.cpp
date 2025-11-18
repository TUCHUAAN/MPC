#include "MPC.h"

MPC::MPC(int n, int r, int m, int N, int a)
{
    this->n = n;
    this->r = r;
    this->m = m;
    this->N = N;
    this->a = a;

    // 系统矩阵初始化为 0，用户外部再设置
    this->A = Eigen::MatrixXd::Zero(this->n, this->n);
    this->B = Eigen::MatrixXd::Zero(this->n, this->r);
    this->C = Eigen::MatrixXd::Zero(this->m, this->n);

    // 权重矩阵初始化为单位阵，防止未设置时出现奇怪的数值
    this->Q = Eigen::MatrixXd::Identity(this->m, this->m);
    this->F = Eigen::MatrixXd::Identity(this->m, this->m);
    this->R = Eigen::MatrixXd::Identity(this->r, this->r);

    // 状态 / 输出 / 参考
    this->x = Eigen::VectorXd::Zero(this->n);            // 当前状态
    this->u = Eigen::VectorXd::Zero(this->r);            // 当前控制输出
    this->y = Eigen::VectorXd::Zero(this->m);            // 输出
    this->ref = Eigen::MatrixXd::Zero(this->m, this->N + 1); // 参考轨迹 m×(N+1)

    // QP 相关矩阵
    this->G = Eigen::MatrixXd::Zero(this->N * this->r, this->N * this->r);
    this->g0 = Eigen::VectorXd::Zero(this->N * this->r);

    // 中间矩阵
    this->Cp = Eigen::MatrixXd::Zero((this->N + 1) * this->n, this->N * this->r);
    this->M = Eigen::MatrixXd::Zero((this->N + 1) * this->n, this->n);
    this->Q1 = Eigen::MatrixXd::Zero((this->N + 1) * this->n, (this->N + 1) * this->n);
    this->Q2 = Eigen::MatrixXd::Zero(1, (this->N + 1) * this->n);
    this->R1 = Eigen::MatrixXd::Zero(this->N * this->r, this->N * this->r);

    // 约束相关（稀疏矩阵内部使用，外部只需要传稠密）
    this->A_q = Eigen::SparseMatrix<double>(this->a, this->r);
    this->lowerBound_q = Eigen::VectorXd::Zero(this->a);
    this->upperBound_q = Eigen::VectorXd::Zero(this->a);

    this->QPSolution = Eigen::VectorXd::Zero(this->N * this->r);
}

// ========================= 外部设置接口实现 =========================

void MPC::setSystemMatrices(const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& C)
{
    if (A.rows() != this->n || A.cols() != this->n)
        throw std::runtime_error("错误：setSystemMatrices 中 A 的维度应为 n×n。");

    if (B.rows() != this->n || B.cols() != this->r)
        throw std::runtime_error("错误：setSystemMatrices 中 B 的维度应为 n×r。");

    if (C.rows() != this->m || C.cols() != this->n)
        throw std::runtime_error("错误：setSystemMatrices 中 C 的维度应为 m×n。");

    this->A = A;
    this->B = B;
    this->C = C;
}


void MPC::setWeightMatrices(const Eigen::VectorXd& Q_vec,
    const Eigen::VectorXd& R_vec,
    const Eigen::VectorXd& F_vec)
{
    // --- 维度检查 ---
    if (Q_vec.size() != this->m)
        throw std::runtime_error("错误：setWeightMatrices 中 Q 向量长度必须为 m。");

    if (F_vec.size() != this->m)
        throw std::runtime_error("错误：setWeightMatrices 中 F 向量长度必须为 m。");

    if (R_vec.size() != this->r)
        throw std::runtime_error("错误：setWeightMatrices 中 R 向量长度必须为 r。");

    // --- 构造对角矩阵 ---
    this->Q = Q_vec.asDiagonal();  // m×m
    this->F = F_vec.asDiagonal();  // m×m
    this->R = R_vec.asDiagonal();  // r×r
}



void MPC::setState(const Eigen::VectorXd& x)
{
    if (x.size() != this->n)
        throw std::runtime_error("错误：setState 中状态向量 x 的长度必须为 n。");

    this->x = x;
}


void MPC::setReference(const Eigen::MatrixXd& ref)
{
    // 情况 1：ref 是 m×(N+1)，完全匹配
    if (ref.rows() == this->m && ref.cols() == this->N + 1)
    {
        this->ref = ref;
        return;
    }

    // 情况 2：ref 是 m×1，自动扩展为 m×(N+1)
    if (ref.rows() == this->m && ref.cols() == 1)
    {
        std::cerr << "警告：setReference 收到的是 m×1 的单列参考，将自动复制为 m×(N+1)。" << std::endl;

        this->ref = Eigen::MatrixXd::Zero(this->m, this->N + 1);

        for (int i = 0; i < this->N + 1; ++i)
        {
            this->ref.col(i) = ref.col(0);
        }
        return;
    }

    // 情况 3：维度完全不匹配 → 抛异常
    throw std::runtime_error("错误：setReference 输入维度错误，应为 m×(N+1) 或 m×1。");
}



void MPC::setInputConstraints(const Eigen::MatrixXd& A_dense,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper)
{
    if (A_dense.rows() != this->a || A_dense.cols() != this->r)
        throw std::runtime_error("错误：setInputConstraints 中 A_dense 的维度必须为 a×r。");

    if (lower.size() != this->a)
        throw std::runtime_error("错误：setInputConstraints 中 lower 的长度必须为 a。");

    if (upper.size() != this->a)
        throw std::runtime_error("错误：setInputConstraints 中 upper 的长度必须为 a。");

    // 转稀疏矩阵
    this->A_q = A_dense.sparseView();
    this->lowerBound_q = lower;
    this->upperBound_q = upper;
}


// ========================= Update：构造 G / g0 =========================

void MPC::Update()
{
    // ---------- 1. 构造 M 和 Cp ----------
    // M: (N+1)*n × n，表示初始状态 x 对未来状态的线性影响
    // Cp: (N+1)*n × N*r，表示未来控制量序列对未来状态的影响

    // temp_A_pow 存 A^k
    Eigen::MatrixXd temp_A_pow = Eigen::MatrixXd::Identity(this->n, this->n);

    // 每一行对应 k=0...N 步的状态
    for (int i = 0; i <= this->N; ++i)
    {
        // 设置 M 的第 i 块 (对应 x_k = A^k x_0 + ...)
        this->M.block(i * this->n, 0, this->n, this->n) = temp_A_pow;

        // 内层循环构造 Cp：
        // 对于固定 i，对应的是“第 i 步”状态，
        // 它受到以往控制量 u_0...u_{i-1} 的影响。
        Eigen::MatrixXd A_power_for_B = temp_A_pow; // A^i

        for (int j = 0; j < this->N; ++j)
        {
            int state_step = i;     // 当前状态步（0...N）
            int input_step = j;     // 当前输入步（0...N-1）

            // 只有当输入步影响得及该状态步时才赋值
            if (input_step <= state_step - 1)
            {
                // 影响次数 = state_step - input_step - 1
                // A^(state_step-1-input_step) * B
                int power = state_step - 1 - input_step;

                // 计算 A^power * B（简单写法：每次从 Identity 开始乘）
                Eigen::MatrixXd A_pow = Eigen::MatrixXd::Identity(this->n, this->n);
                for (int p = 0; p < power; ++p)
                {
                    A_pow = A_pow * this->A;
                }

                Eigen::MatrixXd temp_AB = A_pow * this->B;
                this->Cp.block(state_step * this->n,
                    input_step * this->r,
                    this->n, this->r) = temp_AB;
            }
        }

        // 更新 temp_A_pow = A^{i+1}
        temp_A_pow = temp_A_pow * this->A;
    }

    // ---------- 2. 构造 Q1 ----------
    // Q1 为堆叠后的状态误差权重：x_k 的权重为 C^T Q C，终端步为 C^T F C
    Eigen::MatrixXd temp_Q = this->C.transpose() * this->Q * this->C; // n×n
    Eigen::MatrixXd temp_F = this->C.transpose() * this->F * this->C; // n×n

    this->Q1.setZero();
    for (int i = 0; i <= this->N; ++i)
    {
        if (i == this->N)
        {
            this->Q1.block(i * this->n, i * this->n, this->n, this->n) = temp_F;
        }
        else
        {
            this->Q1.block(i * this->n, i * this->n, this->n, this->n) = temp_Q;
        }
    }

    // ---------- 3. 构造 R1 ----------
    // R1 为堆叠后的输入权重，diag(R, R, ..., R)
    this->R1.setZero();
    for (int i = 0; i < this->N; ++i)
    {
        this->R1.block(i * this->r, i * this->r, this->r, this->r) = this->R;
    }

    // ---------- 4. 构造 G ----------
    // 目标函数：J = U^T G U + g0^T U + 常数
    this->G = 2.0 * (this->Cp.transpose() * this->Q1 * this->Cp + this->R1);

    // ---------- 5. 构造 Q2 ----------
    // 这里 Q2 是 1×((N+1)*n)，与 ref 有关，最后参与 g0 的计算
    // 更合理的写法：
    // 令 e_k = y_ref_k, 则线性项中会有 -2 * e_k^T Q C x_k 等
    this->Q2.setZero();

    // temp_rQ / temp_rF: n×m 矩阵
    Eigen::MatrixXd temp_rQ = -2.0 * this->C.transpose() * this->Q;
    Eigen::MatrixXd temp_rF = -2.0 * this->C.transpose() * this->F;

    for (int i = 0; i <= this->N; ++i)
    {
        // 参考输出 e_k: m×1
        Eigen::VectorXd ref_k = this->ref.col(i); // m×1

        if (i == this->N)
        {
            // 1×n = (1×m) * (m×n)
            Eigen::RowVectorXd row = ref_k.transpose() * temp_rF;
            this->Q2.block(0, i * this->n, 1, this->n) = row;
        }
        else
        {
            Eigen::RowVectorXd row = ref_k.transpose() * temp_rQ;
            this->Q2.block(0, i * this->n, 1, this->n) = row;
        }
    }

    // ---------- 6. 构造 g0 ----------
    // g0^T = 2 * (M x)^T Q1 Cp + Q2 Cp
    // 先算一个 1×(N*r) 的行向量，再转成列向量
    Eigen::RowVectorXd row_g0 =
        2.0 * (this->x.transpose() * this->M.transpose() * this->Q1 * this->Cp)
        + this->Q2 * this->Cp;

    this->g0 = row_g0.transpose(); // N*r × 1
}

// ========================= QP 求解 =========================

bool MPC::Solve_OsqpEigen()
{
    // ---------- 1. 构造 Hessian 和 gradient ----------
    Eigen::SparseMatrix<double> hessian(this->N * this->r, this->N * this->r);
    hessian = this->G.sparseView();  // 直接从稠密矩阵转换为稀疏矩阵

    Eigen::VectorXd gradient = this->g0; // N*r × 1

    // ---------- 2. 构造约束矩阵 ----------
    // 整个预测区间的约束：对每一时刻重复 A_q, lowerBound_q, upperBound_q
    int rowsPerStep = this->A_q.rows();
    int totalRows = this->N * rowsPerStep;

    Eigen::SparseMatrix<double> linearMatrix(totalRows, this->N * this->r);
    Eigen::VectorXd lowerBound(totalRows);
    Eigen::VectorXd upperBound(totalRows);

    // 复制 A_q 到对角块（实际不是严格块对角，只是每步约束作用在对应的 u_k 上）
    for (int k = 0; k < this->N; ++k)
    {
        for (int i = 0; i < rowsPerStep; ++i)
        {
            for (int j = 0; j < this->r; ++j)
            {
                double val = this->A_q.coeff(i, j);
                if (val != 0.0)
                {
                    linearMatrix.insert(k * rowsPerStep + i,
                        k * this->r + j) = val;
                }
            }
        }
    }

    // 下上界复制 N 次
    for (int k = 0; k < this->N; ++k)
    {
        for (int i = 0; i < rowsPerStep; ++i)
        {
            lowerBound[k * rowsPerStep + i] = this->lowerBound_q(i);
            upperBound[k * rowsPerStep + i] = this->upperBound_q(i);
        }
    }

    // ---------- 3. OSQP 求解 ----------
    OsqpEigen::Solver solver;

    // 设置求解器参数
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    // 可视情况打开下面这些来调性能 / 收敛：
    // solver.settings()->setAbsoluteTolerance(1e-3);
    // solver.settings()->setRelativeTolerance(1e-3);
    // solver.settings()->setMaxIteration(200);

    // 设置 QP 维度
    solver.data()->setNumberOfVariables(this->N * this->r);
    solver.data()->setNumberOfConstraints(totalRows);

    if (!solver.data()->setHessianMatrix(hessian))             return false;
    if (!solver.data()->setGradient(gradient))                 return false;
    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return false;
    if (!solver.data()->setLowerBound(lowerBound))             return false;
    if (!solver.data()->setUpperBound(upperBound))             return false;

    if (!solver.initSolver()) return false;

    // 求解
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cerr << "[MPC::Solve_OsqpEigen] OSQP 求解失败" << std::endl;
        return false;
    }

    this->QPSolution = solver.getSolution(); // N*r × 1

    // 取第一步控制量作为当前输出 u
    for (int i = 0; i < this->r; ++i)
    {
        this->u(i) = this->QPSolution(i);
    }

    return true;
}
