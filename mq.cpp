// Maldacena-Qi モデル: 相互作用する 2 つの SYK (L, R) の厳密シミュレーション
//
// H = H_L + H_R + i mu sum_j chi_L^j chi_R^j      (q=4, 各側に N 個の Majorana)
//   H_L = sum J_abcd chi_L^a chi_L^b chi_L^c chi_L^d,  H_R は同じ J(符号は TFD 条件から決まる)
//   <J^2> = 3! J^2 / N^3
//
// Jordan-Wigner では L_j = chi_{2j}, R_j = chi_{2j+1} と並べる。すると
//   i chi_L^j chi_R^j = -Z_j/2  なので、相互作用項は対角行列 -(mu/2) sum_j Z_j。
//   |I> = |00..0> は (chi_L + i chi_R)|I> = 0 を満たす無限温度 TFD。
//
// 計算するもの(偶パリティ部分空間):
//   * Lanczos 法による基底・低励起エネルギー(ギャップ)
//   * 基底状態と TFD 状態 exp(-beta H_L/2)|I> の重なり(beta 最適化)
//   * <i chi_L chi_R> 相関
//   * 実時間相関 G_LL(t), G_LR(t) (オプション)
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstdint>
using namespace std;
using cplx = complex<double>;
using CVec = vector<cplx>;

const int MAX_N = 12;          // 片側の Majorana 数の上限(次元 2^N)
const int N_LEVELS = 4;        // 出力する励起準位数
const int LANCZOS_STEPS = 150;

static inline uint32_t apply_chi(int a, uint32_t s, cplx &phase) {
    int k = a / 2;
    int sign = (__builtin_popcount(s & ((1u << k) - 1)) & 1) ? -1 : 1;
    int bit = (s >> k) & 1;
    cplx f = sign / sqrt(2.0);
    if (a % 2 == 1) f *= bit ? cplx(0, -1) : cplx(0, 1);
    phase *= f;
    return s ^ (1u << k);
}

// 列ごとに格納した疎行列: y[col] += val * x[s]
struct Sparse {
    vector<int> ptr, row;
    vector<cplx> val;
    void apply(const CVec &x, CVec &y, bool accumulate = false) const {
        if (!accumulate) fill(y.begin(), y.end(), cplx(0));
        for (size_t s = 0; s + 1 < ptr.size(); ++s) {
            if (x[s] == cplx(0)) continue;
            for (int k = ptr[s]; k < ptr[s + 1]; ++k) y[row[k]] += val[k] * x[s];
        }
    }
};

struct Quartet { int a, b, c, d; double J; };

// side=0: L (chi_{2j}), side=1: R (chi_{2j+1})。sgn は R 側の符号。
Sparse build_sparse(int N, const vector<Quartet> &qs, int side, double sgn) {
    int dim = 1 << N;
    Sparse S;
    S.ptr.assign(dim + 1, 0);
    vector<pair<int, cplx>> col;
    for (int s = 0; s < dim; ++s) {
        col.clear();
        for (const Quartet &q : qs) {
            cplx ph = sgn * q.J;
            uint32_t t = s;
            t = apply_chi(2 * q.d + side, t, ph);
            t = apply_chi(2 * q.c + side, t, ph);
            t = apply_chi(2 * q.b + side, t, ph);
            t = apply_chi(2 * q.a + side, t, ph);
            col.push_back({(int)t, ph});
        }
        sort(col.begin(), col.end(), [](auto &x, auto &y) { return x.first < y.first; });
        for (size_t i = 0; i < col.size();) {
            size_t j = i; cplx v = 0;
            while (j < col.size() && col[j].first == col[i].first) v += col[j++].second;
            S.row.push_back(col[i].first);
            S.val.push_back(v);
            i = j;
        }
        S.ptr[s + 1] = S.row.size();
    }
    return S;
}

cplx dotc(const CVec &a, const CVec &b) {  // <a|b>
    cplx s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += conj(a[i]) * b[i];
    return s;
}
double norm2(const CVec &a) { return sqrt(dotc(a, a).real()); }

// 小さな実対称行列のヤコビ法(Lanczos の三重対角行列用)。Vt の行 k が固有ベクトル。
void jacobi_small(vector<vector<double>> A, vector<double> &d, vector<vector<double>> &Vt) {
    int n = A.size();
    Vt.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Vt[i][i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0, diag = 0;
        for (int i = 0; i < n; ++i) {
            diag += A[i][i] * A[i][i];
            for (int j = i + 1; j < n; ++j) off += A[i][j] * A[i][j];
        }
        if (off < 1e-28 * (diag + off)) break;
        for (int p = 0; p < n - 1; ++p)
        for (int q = p + 1; q < n; ++q) {
            double apq = A[p][q];
            if (fabs(apq) < 1e-300) continue;
            double tau = (A[q][q] - A[p][p]) / (2.0 * apq);
            double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
            double c = 1.0 / sqrt(1.0 + t * t), sn = t * c;
            for (int k = 0; k < n; ++k) {
                if (k == p || k == q) continue;
                double akp = A[k][p], akq = A[k][q];
                A[k][p] = A[p][k] = c * akp - sn * akq;
                A[k][q] = A[q][k] = sn * akp + c * akq;
            }
            A[p][p] -= t * apq; A[q][q] += t * apq; A[p][q] = A[q][p] = 0.0;
            for (int k = 0; k < n; ++k) {
                double vp = Vt[p][k], vq = Vt[q][k];
                Vt[p][k] = c * vp - sn * vq;
                Vt[q][k] = sn * vp + c * vq;
            }
        }
    }
    d.resize(n);
    for (int i = 0; i < n; ++i) d[i] = A[i][i];
}

struct Model {
    int N, dim;
    Sparse HL, HR;
    vector<double> zsum;  // sum_j Z_j (各基底状態)
    double mu = 0;
    // y = H x
    void apply(const CVec &x, CVec &y) const {
        HL.apply(x, y);
        HR.apply(x, y, true);
        for (int s = 0; s < dim; ++s) y[s] += (-0.5 * mu * zsum[s]) * x[s];
    }
};

// 完全再直交化つき Lanczos。偶パリティ部分空間の最低 nlev 個のエネルギーと基底状態。
void lanczos(const Model &M, mt19937_64 &rng, int nlev, vector<double> &E, CVec &gs) {
    int dim = M.dim;
    normal_distribution<double> g(0, 1);
    CVec v(dim, 0.0);
    for (int s = 0; s < dim; ++s)
        if (!(__builtin_popcount(s) & 1)) v[s] = cplx(g(rng), g(rng));
    double nr = norm2(v);
    for (auto &x : v) x /= nr;

    int m = min(LANCZOS_STEPS, dim / 2);
    vector<CVec> Q;
    vector<double> alpha, beta;
    CVec w(dim);
    Q.push_back(v);
    for (int k = 0; k < m; ++k) {
        M.apply(Q[k], w);
        alpha.push_back(dotc(Q[k], w).real());
        for (int pass = 0; pass < 2; ++pass)
            for (auto &q : Q) {
                cplx c = dotc(q, w);
                for (int s = 0; s < dim; ++s) w[s] -= c * q[s];
            }
        double b = norm2(w);
        if (b < 1e-10 || k == m - 1) break;
        beta.push_back(b);
        CVec nq(dim);
        for (int s = 0; s < dim; ++s) nq[s] = w[s] / b;
        Q.push_back(nq);
    }
    int n = alpha.size();
    vector<vector<double>> T(n, vector<double>(n, 0.0)), Vt;
    for (int i = 0; i < n; ++i) {
        T[i][i] = alpha[i];
        if (i + 1 < n && i < (int)beta.size()) T[i][i + 1] = T[i + 1][i] = beta[i];
    }
    vector<double> d;
    jacobi_small(T, d, Vt);
    vector<int> ord(n);
    for (int i = 0; i < n; ++i) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int a, int b) { return d[a] < d[b]; });
    E.clear();
    for (int i = 0; i < min(nlev, n); ++i) E.push_back(d[ord[i]]);
    gs.assign(dim, 0.0);
    for (int k = 0; k < n; ++k)
        for (int s = 0; s < dim; ++s) gs[s] += Vt[ord[0]][k] * Q[k][s];
    double gn = norm2(gs);
    for (auto &x : gs) x /= gn;
}

// Taylor 展開で x <- exp(-z * H) x  (z は複素: 実時間なら z = i dt)
template <class Op>
void taylor_step(const Op &H, CVec &x, cplx z, int order) {
    CVec term = x, tmp(x.size());
    for (int k = 1; k <= order; ++k) {
        H(term, tmp);
        for (size_t s = 0; s < x.size(); ++s) term[s] = -z * tmp[s] / double(k);
        for (size_t s = 0; s < x.size(); ++s) x[s] += term[s];
    }
}

CVec apply_majorana(int a, const CVec &x) {  // chi_a x
    CVec y(x.size(), 0.0);
    for (size_t s = 0; s < x.size(); ++s) {
        if (x[s] == cplx(0)) continue;
        cplx ph = 1.0;
        uint32_t t = apply_chi(a, s, ph);
        y[t] += ph * x[s];
    }
    return y;
}

struct Result {
    vector<double> levels;      // E_k - E_0
    double e0 = 0, zavg = 0, overlap = 0, beta_star = 0;
    vector<cplx> gll, glr;      // 実時間相関
};

int main(int argc, char *argv[]) {
    if (argc < 6) {
        cout << "Usage: " << argv[0] << " <N> <samples> <seed> <tmax> <mu1> [mu2 ...]\n"
             << "  N: 片側の Majorana 数 (偶数, 4.." << MAX_N << ", 次元 2^N)\n"
             << "  tmax>0 で実時間相関 G_LL(t), G_LR(t) も計算 (J=1)\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int samples = atoi(argv[2]);
    unsigned long seed = strtoul(argv[3], nullptr, 10);
    double tmax = atof(argv[4]);
    vector<double> mus;
    for (int i = 5; i < argc; ++i) mus.push_back(atof(argv[i]));
    if (N < 4 || N > MAX_N || N % 2 || samples < 1) {
        cerr << "Error: N は 4.." << MAX_N << " の偶数、samples >= 1\n";
        return 1;
    }

    const int dim = 1 << N;
    const double BETA_STEP = 0.5, BETA_MAX = 40.0;
    const double DT = 0.1;
    const int ntime = tmax > 0 ? (int)round(tmax / DT) : 0;
    const int nj = min(N, 4);  // 実時間相関で平均する flavor 数

    vector<Result> avg(mus.size());
    for (auto &r : avg) { r.levels.assign(N_LEVELS, 0.0); r.gll.assign(ntime + 1, 0.0); r.glr.assign(ntime + 1, 0.0); }

    for (int smp = 0; smp < samples; ++smp) {
        mt19937_64 rng(seed + 1000003ul * smp);
        normal_distribution<double> gauss(0.0, sqrt(6.0 / ((double)N * N * N)));
        vector<Quartet> qs;
        for (int a = 0; a < N; ++a) for (int b = a + 1; b < N; ++b)
        for (int c = b + 1; c < N; ++c) for (int d = c + 1; d < N; ++d)
            qs.push_back({a, b, c, d, gauss(rng)});

        Model M;
        M.N = N; M.dim = dim;
        M.HL = build_sparse(N, qs, 0, 1.0);
        M.HR = build_sparse(N, qs, 1, 1.0);
        M.zsum.assign(dim, 0.0);
        for (int s = 0; s < dim; ++s)
            for (int j = 0; j < N; ++j) M.zsum[s] += ((s >> j) & 1) ? -1.0 : 1.0;

        // TFD 条件 H_L|I> = H_R|I> の確認(R 側の符号を決める)
        CVec I(dim, 0.0), x(dim), y(dim);
        I[0] = 1.0;
        M.HL.apply(I, x); M.HR.apply(I, y);
        double dp = 0, dm = 0;
        for (int s = 0; s < dim; ++s) { dp += norm(x[s] - y[s]); dm += norm(x[s] + y[s]); }
        if (dp > 1e-18) {
            if (dm > 1e-18) { cerr << "Error: H_L|I> != +-H_R|I>\n"; return 1; }
            for (auto &v : M.HR.val) v = -v;  // J_R = -J_L
        }

        // TFD 状態の列 exp(-beta H_L/2)|I> を保存
        vector<CVec> tfd;
        {
            CVec v = I;
            auto HLop = [&](const CVec &a, CVec &b) { M.HL.apply(a, b); };
            tfd.push_back(v);
            for (double beta = BETA_STEP; beta <= BETA_MAX + 1e-9; beta += BETA_STEP) {
                taylor_step(HLop, v, cplx(BETA_STEP / 2), 14);
                double n = norm2(v);
                for (auto &c : v) c /= n;
                tfd.push_back(v);
            }
        }

        for (size_t im = 0; im < mus.size(); ++im) {
            M.mu = mus[im];
            vector<double> E; CVec gs;
            lanczos(M, rng, N_LEVELS + 1, E, gs);
            Result &r = avg[im];
            r.e0 += E[0];
            for (int k = 0; k < N_LEVELS && k + 1 < (int)E.size(); ++k) r.levels[k] += E[k + 1] - E[0];
            // <Z> 平均
            double z = 0;
            for (int s = 0; s < dim; ++s) z += norm(gs[s]) * M.zsum[s] / N;
            r.zavg += z;
            // TFD 重なり
            double best = 0, bstar = 0;
            for (size_t k = 0; k < tfd.size(); ++k) {
                double ov = norm(dotc(gs, tfd[k]));
                if (ov > best) { best = ov; bstar = k * BETA_STEP; }
            }
            r.overlap += best; r.beta_star += bstar;

            // 実時間相関 G(t) = e^{iE0 t} <chi gs| e^{-iHt} |chi gs>
            if (ntime > 0) {
                auto Hop = [&](const CVec &a, CVec &b) { M.apply(a, b); };
                for (int j = 0; j < nj; ++j) {
                    CVec phiL = apply_majorana(2 * j, gs), phiR = apply_majorana(2 * j + 1, gs);
                    CVec uL = phiL, uR = phiR;
                    for (int it = 0; it <= ntime; ++it) {
                        cplx ph = exp(cplx(0, E[0] * it * DT));
                        r.gll[it] += ph * dotc(phiL, uL) / double(nj);
                        r.glr[it] += ph * dotc(phiL, uR) / double(nj);
                        if (it < ntime) {
                            taylor_step(Hop, uL, cplx(0, DT), 12);
                            taylor_step(Hop, uR, cplx(0, DT), 12);
                        }
                    }
                }
            }
        }
    }

    cout << fixed << setprecision(5);
    cout << "# Maldacena-Qi: N=" << N << " (per side), samples=" << samples << ", seed=" << seed << ", J=1\n";
    cout << "# mu   E0/N   gap(E1-E0)  E2-E0  E3-E0  E4-E0   <Z>   max|<TFD|GS>|^2   beta*\n";
    for (size_t im = 0; im < mus.size(); ++im) {
        const Result &r = avg[im];
        double inv = 1.0 / samples;
        cout << mus[im] << "  " << r.e0 * inv / N;
        for (int k = 0; k < N_LEVELS; ++k) cout << "  " << r.levels[k] * inv;
        cout << "  " << r.zavg * inv << "  " << r.overlap * inv << "  " << r.beta_star * inv << "\n";
    }
    if (ntime > 0) {
        for (size_t im = 0; im < mus.size(); ++im) {
            cout << "\n# mu=" << mus[im] << ": t  Re G_LL  Im G_LL  Re G_LR  Im G_LR  (G(0)=1/2)\n";
            for (int it = 0; it <= ntime; ++it) {
                double inv = 1.0 / samples;
                cout << it * DT << "  " << avg[im].gll[it].real() * inv << "  " << avg[im].gll[it].imag() * inv
                     << "  " << avg[im].glr[it].real() * inv << "  " << avg[im].glr[it].imag() * inv << "\n";
            }
        }
    }
    return 0;
}
