#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>
#include <cstdlib>
#include <Eigen/Dense>
using namespace std;
using Vec = vector<double>;

// 定数定義
const double EPSILON = 1e-16;        // 数値精度の閾値
const double CONVERGENCE_THRESHOLD = 1e-12;  // 収束判定の閾値
const double GAP_TOLERANCE = 1e-8;   // ギャップ収束の許容誤差

// 内積
double dot(const Vec &a, const Vec &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// 正規化
void normalize(Vec &v) {
    double n = sqrt(dot(v, v));
    if (n < EPSILON) return;
    for (double &x : v) x /= n;
}

// スピン1/2 チェインの H を作用させる: out = H * in
void apply_H_half(const Vec &in, Vec &out, int L, double J = 1.0, bool periodic = false) {
    int dim = 1 << L;
    fill(out.begin(), out.end(), 0.0);

    for (int s = 0; s < dim; ++s) {
        double amp = in[s];
        if (fabs(amp) < EPSILON) continue;
        for (int i = 0; i < L - 1 + (periodic ? 1 : 0); ++i) {
            int j = (i + 1) % L;
            // bit=1 -> up (m=+1/2), bit=0 -> down (m=-1/2)
            int bi = (s >> i) & 1;
            int bj = (s >> j) & 1;
            double mi = bi ? 0.5 : -0.5;
            double mj = bj ? 0.5 : -0.5;

            // Sz Sz term
            out[s] += J * mi * mj * amp;

            // Flip terms: S_i^+ S_j^- and S_i^- S_j^+
            if (bi == 0 && bj == 1) {
                // S_i^+ S_j^-
                int s2 = s ^ (1 << i) ^ (1 << j);
                out[s2] += 0.5 * J * amp;
            }
            if (bi == 1 && bj == 0) {
                // S_i^- S_j^+
                int s2 = s ^ (1 << i) ^ (1 << j);
                out[s2] += 0.5 * J * amp;
            }
        }
    }
}

// スピン1 チェインの H を作用させる: out = H * in
// 状態は base-3 表現（各サイト: m = -1,0,1 を 0,1,2 で符号化）
void apply_H_one(const Vec &in, Vec &out, int L, double J = 1.0, bool periodic = false) {
    int dim = 1;
    for (int i = 0; i < L; ++i) dim *= 3;
    fill(out.begin(), out.end(), 0.0);

    // 事前計算：各サイトの係数
    vector<int> pow3(L);
    pow3[0] = 1;
    for (int i = 1; i < L; ++i) pow3[i] = pow3[i-1] * 3;

    for (int idx = 0; idx < dim; ++idx) {
        double amp = in[idx];
        if (fabs(amp) < EPSILON) continue;

        // 各サイトの m 値を展開（-1,0,1）
        vector<int> m(L);
        int tmp = idx;
        for (int i = 0; i < L; ++i) {
            int d = tmp % 3;
            m[i] = d - 1; // 0->-1,1->0,2->1
            tmp /= 3;
        }

        for (int i = 0; i < L - 1 + (periodic ? 1 : 0); ++i) {
            int j = (i + 1) % L;
            int mi = m[i];
            int mj = m[j];

            // Sz Sz term
            out[idx] += J * (double)mi * (double)mj * amp;

            // S_i^+ S_j^- term: m_i -> m_i+1, m_j -> m_j-1
            if (mi < 1 && mj > -1) {
                double coef_i = sqrt(1.0 * 2.0 - mi * (mi + 1));
                double coef_j = sqrt(1.0 * 2.0 - mj * (mj - 1));
                double coeff = 0.5 * J * coef_i * coef_j;

                int new_idx = idx + pow3[i] * 1 + pow3[j] * (-1); // m[i]+1, m[j]-1
                out[new_idx] += coeff * amp;
            }
            // S_i^- S_j^+ term: m_i -> m_i-1, m_j -> m_j+1
            if (mi > -1 && mj < 1) {
                double coef_i = sqrt(1.0 * 2.0 - mi * (mi - 1));
                double coef_j = sqrt(1.0 * 2.0 - mj * (mj + 1));
                double coeff = 0.5 * J * coef_i * coef_j;

                int new_idx = idx + pow3[i] * (-1) + pow3[j] * 1; // m[i]-1, m[j]+1
                out[new_idx] += coeff * amp;
            }
        }
    }
}

// Lanczos 再帰（対象的対称実行行列）
// apply_H_func: (in, out) を受け取り out := H * in
// max_it: 最大反復
// returns 最小2固有値を pair (E0, E1)
using ApplyFunc = function<void(const Vec&, Vec&)>;

pair<double,double> lanczos_two_lowest(int dim, ApplyFunc apply_H_func, int max_it = 100) {
    vector<double> alpha;
    vector<double> beta;
    vector<Vec> vs; // basis vectors v_0, v_1, ...

    Vec v(dim), w(dim), v_prev(dim);
    // 初期ベクトルを乱数で用意して正規化
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int i = 0; i < dim; ++i) v[i] = uni(rng);
    normalize(v);
    vs.push_back(v);
    double beta_prev = 0.0;

    for (int k = 0; k < max_it; ++k) {
        apply_H_func(v, w); // w = H v

        double a = dot(v, w);
        alpha.push_back(a);

        if (k == 0) {
            // w = w - a v
            for (int i = 0; i < dim; ++i) w[i] -= a * v[i];
        } else {
            // w = w - a v - beta_prev * v_prev
            for (int i = 0; i < dim; ++i) w[i] -= a * v[i] + beta_prev * v_prev[i];
        }

        // 再正規直交
        double b = sqrt(dot(w, w));
        if (b < CONVERGENCE_THRESHOLD) break;

        beta.push_back(b);
        // 準備次回
        v_prev = v;
        for (int i = 0; i < dim; ++i) v[i] = w[i] / b;
        vs.push_back(v);
        beta_prev = b;

        // トリジナル三重対角行列を構成して対角化
        int m = alpha.size();
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i) {
            T(i,i) = alpha[i];
            if (i + 1 < m) {
                T(i, i+1) = beta[i];
                T(i+1, i) = beta[i];
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
        if (es.info() != Eigen::Success) continue;
        Eigen::VectorXd evals = es.eigenvalues();
        // 最小2つを取る（Ritz 値）
        double E0 = evals[0];
        double E1 = (m >= 2 ? evals[1] : evals[0]);
        
        // 収束判定：ギャップの変化が小さい場合に早期終了
        if (k > 5) {
            static double prev_gap = 0.0;
            double current_gap = E1 - E0;
            if (fabs(current_gap - prev_gap) < GAP_TOLERANCE) {
                break;  // 収束したと判断して終了
            }
            prev_gap = current_gap;
        }
    }

    // 最終的な三重対角 T を対角化（長さ alpha.size()）
    int m = alpha.size();
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < m; ++i) {
        T(i,i) = alpha[i];
        if (i + 1 < m) {
            T(i, i+1) = beta[i];
            T(i+1, i) = beta[i];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    Eigen::VectorXd evals = es.eigenvalues();
    double E0 = evals[0];
    double E1 = (m >= 2 ? evals[1] : evals[0]);
    return {E0, E1};
}

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Usage: ./haldane <spin: 0.5 or 1> <L_max> <lanczos_steps> [periodic=0/1] [L_min=4]\n";
        cout << "Example: ./haldane 0.5 12 50 0 4  # Spin-1/2, L=4 to 12, 50 Lanczos steps, OBC\n";
        cout << "Note: Lanczos algorithm now includes convergence checking for early termination.\n";
        return 1;
    }
    double spin = atof(argv[1]);
    int L_max = atoi(argv[2]);
    int max_it = atoi(argv[3]);
    bool periodic = false;
    if (argc >= 5) periodic = atoi(argv[4]) != 0;
    int L_min = 4;
    if (argc >= 6) L_min = atoi(argv[5]);

    double J = 1.0;

    cout << "# Scaling analysis for Spin-" << spin << " chain\n";
    cout << "# L_min=" << L_min << ", L_max=" << L_max << ", Lanczos_steps=" << max_it << "\n";
    cout << "# Boundary: " << (periodic ? "PBC" : "OBC") << "\n";
    cout << "# L\tGround_E\t1st_excited_E\tGap\tE_per_site\tGap_per_site\n";

    for (int L = L_min; L <= L_max; ++L) {
        if (abs(spin - 0.5) < 1e-9) {
            int dim = 1 << L;
            Vec v(dim), out(dim);
            auto [E0, E1] = lanczos_two_lowest(dim, [&](const Vec &in, Vec &o){ apply_H_half(in, o, L, J, periodic); }, max_it);
            double gap = E1 - E0;
            double E_per_site = E0 / L;
            double gap_per_site = gap / L;
            cout << L << "\t" << E0 << "\t" << E1 << "\t" << gap << "\t" << E_per_site << "\t" << gap_per_site << "\n";
        } else if (abs(spin - 1.0) < 1e-9) {
            int dim = 1;
            for (int i = 0; i < L; ++i) dim *= 3;
            Vec v(dim), out(dim);
            auto [E0, E1] = lanczos_two_lowest(dim, [&](const Vec &in, Vec &o){ apply_H_one(in, o, L, J, periodic); }, max_it);
            double gap = E1 - E0;
            double E_per_site = E0 / L;
            double gap_per_site = gap / L;
            cout << L << "\t" << E0 << "\t" << E1 << "\t" << gap << "\t" << E_per_site << "\t" << gap_per_site << "\n";
        } else {
            cerr << "Error: Unsupported spin value " << spin << ". Only 0.5 and 1 are implemented.\n";
            cerr << "Please use 0.5 for spin-1/2 chains or 1.0 for spin-1 chains.\n";
            return 1;
        }
    }
    return 0;
}
