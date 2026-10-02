// SYK (Sachdev-Ye-Kitaev) モデルの厳密対角化シミュレーション
//
// H = sum_{a<b<c<d} J_{abcd} chi_a chi_b chi_c chi_d
//   {chi_a, chi_b} = delta_ab,  <J_{abcd}^2> = 3! J^2 / N^3   (N 個の Majorana)
//
// Jordan-Wigner 変換で N/2 個のスピン(2^(N/2) 次元)に写し、
// フェルミオンパリティが偶の部分空間(次元 2^(N/2-1))で対角化する。
// エルミート行列 A+iB は実対称行列 [[A,-B],[B,A]] に埋め込んで解く(固有値は2重縮退)。
// 外部ライブラリ不要。
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
using Mat = vector<vector<double>>;

const double DEGENERACY_TOL = 1e-8;  // 縮退とみなす固有値差
const double PI = 3.14159265358979323846;

// Majorana 演算子 chi_a を基底状態 s に作用させる。
// a=2k: (Z..Z X_k)/sqrt2,  a=2k+1: (Z..Z Y_k)/sqrt2
// 戻り値: 新しい状態。phase に係数が掛かる。
static inline uint32_t apply_chi(int a, uint32_t s, cplx &phase) {
    int k = a / 2;
    int sign = __builtin_popcount(s & ((1u << k) - 1)) & 1 ? -1 : 1;  // 下位サイトの Z
    int bit = (s >> k) & 1;
    cplx f = sign / sqrt(2.0);
    if (a % 2 == 1) f *= bit ? cplx(0, -1) : cplx(0, 1);  // Y|1>=-i|0>, Y|0>=i|1>
    phase *= f;
    return s ^ (1u << k);
}

// 実対称行列の固有値のみ(Householder 三重対角化 + 陰的QL法)
void symmetric_eigenvalues(Mat a, vector<double> &d) {
    int n = a.size();
    d.assign(n, 0.0);
    vector<double> e(n, 0.0), v(n), p(n), w(n);
    for (int k = 0; k + 2 < n; ++k) {
        double nrm = 0;
        for (int i = k + 1; i < n; ++i) nrm += a[i][k] * a[i][k];
        nrm = sqrt(nrm);
        double x0 = a[k + 1][k];
        double alpha = x0 > 0 ? -nrm : nrm;
        e[k] = alpha;
        if (nrm < 1e-300) continue;
        double vn = 0;
        for (int i = k + 1; i < n; ++i) {
            v[i] = a[i][k] - (i == k + 1 ? alpha : 0.0);
            vn += v[i] * v[i];
        }
        vn = sqrt(vn);
        if (vn < 1e-300) continue;
        for (int i = k + 1; i < n; ++i) v[i] /= vn;
        // A -= 2(v w^T + w v^T),  p = A v,  w = p - (v.p) v  (trailing block)
        double beta = 0;
        for (int i = k + 1; i < n; ++i) {
            double s = 0;
            for (int j = k + 1; j < n; ++j) s += a[i][j] * v[j];
            p[i] = s;
            beta += v[i] * s;
        }
        for (int i = k + 1; i < n; ++i) w[i] = p[i] - beta * v[i];
        for (int i = k + 1; i < n; ++i)
            for (int j = k + 1; j < n; ++j)
                a[i][j] -= 2 * (v[i] * w[j] + w[i] * v[j]);
        a[k + 1][k] = a[k][k + 1] = alpha;
        for (int i = k + 2; i < n; ++i) a[i][k] = a[k][i] = 0.0;
    }
    for (int i = 0; i < n; ++i) d[i] = a[i][i];
    if (n >= 2) e[n - 2] = a[n - 1][n - 2];
    e[n - 1] = 0.0;

    // QL 法
    for (int l = 0; l < n; ++l) {
        int iter = 0, m;
        do {
            for (m = l; m < n - 1; ++m) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= 1e-16 * dd) break;
            }
            if (m != l) {
                if (++iter > 200) { cerr << "QL did not converge\n"; exit(1); }
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0 ? fabs(r) : -fabs(r)));
                double s = 1, c = 1, pp = 0;
                int i;
                for (i = m - 1; i >= l; --i) {
                    double f = s * e[i], b = c * e[i];
                    e[i + 1] = r = hypot(f, g);
                    if (r == 0.0) { d[i + 1] -= pp; e[m] = 0.0; break; }
                    s = f / r; c = g / r;
                    g = d[i + 1] - pp;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (pp = s * r);
                    g = c * r - b;
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= pp; e[l] = g; e[m] = 0.0;
            }
        } while (m != l);
    }
    sort(d.begin(), d.end());
}

// SYK のランダム実現1つを作り、偶パリティ部分空間の固有値(縮退除去済み)を返す
vector<double> syk_spectrum(int N, double J, mt19937_64 &rng) {
    int nsp = N / 2;
    vector<uint32_t> basis;
    vector<int> index(1u << nsp, -1);
    for (uint32_t s = 0; s < (1u << nsp); ++s)
        if (!(__builtin_popcount(s) & 1)) { index[s] = basis.size(); basis.push_back(s); }
    int n = basis.size();

    double sigma = sqrt(6.0 * J * J / ((double)N * N * N));
    normal_distribution<double> gauss(0.0, sigma);
    vector<vector<cplx>> H(n, vector<cplx>(n, 0.0));

    for (int a = 0; a < N; ++a)
    for (int b = a + 1; b < N; ++b)
    for (int c = b + 1; c < N; ++c)
    for (int d = c + 1; d < N; ++d) {
        double Jabcd = gauss(rng);
        for (int i = 0; i < n; ++i) {
            cplx ph = Jabcd;
            uint32_t s = basis[i];
            s = apply_chi(d, s, ph);  // chi_a chi_b chi_c chi_d |s>: 右から順に作用
            s = apply_chi(c, s, ph);
            s = apply_chi(b, s, ph);
            s = apply_chi(a, s, ph);
            H[index[s]][i] += ph;
        }
    }

    // 実対称埋め込み
    Mat R(2 * n, vector<double>(2 * n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double re = H[i][j].real(), im = H[i][j].imag();
            R[i][j] = R[i + n][j + n] = re;
            R[i + n][j] = im;
            R[i][j + n] = -im;
        }
    vector<double> ev;
    symmetric_eigenvalues(R, ev);

    vector<double> out;  // 2重縮退を除き、さらに Kramers 縮退(GSE)もまとめる
    for (size_t i = 0; i < ev.size(); i += 2) {
        if (out.empty() || ev[i] - out.back() > DEGENERACY_TOL) out.push_back(ev[i]);
    }
    return out;
}

// 隣接準位間隔比 <r> (GOE≈0.5307, GUE≈0.5996, GSE≈0.6744)
double spacing_ratio(const vector<double> &E) {
    // スペクトル端を避け中央 1/2 のみ使用
    size_t lo = E.size() / 4, hi = E.size() * 3 / 4;
    double sum = 0; int cnt = 0;
    for (size_t i = lo + 1; i + 1 < hi; ++i) {
        double s1 = E[i] - E[i - 1], s2 = E[i + 1] - E[i];
        if (max(s1, s2) < 1e-14) continue;
        sum += min(s1, s2) / max(s1, s2);
        ++cnt;
    }
    return cnt ? sum / cnt : 0.0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <N> [samples=10] [seed=1] [J=1] [dump_spectrum=0]\n"
             << "  N: Majorana 数 (偶数, 4..24)\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int samples = argc > 2 ? atoi(argv[2]) : 10;
    unsigned long seed = argc > 3 ? strtoul(argv[3], nullptr, 10) : 1;
    double J = argc > 4 ? atof(argv[4]) : 1.0;
    bool dump = argc > 5 && atoi(argv[5]) != 0;
    if (N < 4 || N > 24 || N % 2 != 0 || samples < 1) {
        cerr << "Error: N は 4..24 の偶数、samples >= 1 にしてください\n";
        return 1;
    }

    mt19937_64 rng(seed);
    double sumE0 = 0, sumE0sq = 0, sumR = 0, sumGap = 0, sumBand = 0;
    for (int s = 0; s < samples; ++s) {
        vector<double> E = syk_spectrum(N, J, rng);
        double E0 = E.front();
        sumE0 += E0; sumE0sq += E0 * E0;
        sumGap += E[1] - E[0];
        sumBand += E.back() - E.front();
        sumR += spacing_ratio(E);
        if (dump && s == 0) {
            cout << "# spectrum (sample 0, even parity, degeneracy removed)\n";
            cout << setprecision(10);
            for (double x : E) cout << x << "\n";
        }
    }
    double m = sumE0 / samples;
    double err = samples > 1 ? sqrt(max(0.0, sumE0sq / samples - m * m) / (samples - 1)) : 0.0;
    cout << fixed << setprecision(6);
    cout << "N=" << N << "  samples=" << samples << "  J=" << J << "  seed=" << seed << "\n";
    cout << "E0/N        = " << m / N << " +- " << err / N << "   (N->inf, q=4: ~ -0.0406 J)\n";
    cout << "gap(E1-E0)  = " << sumGap / samples << "\n";
    cout << "bandwidth   = " << sumBand / samples << "\n";
    cout << "<r>         = " << sumR / samples
         << "   (GOE 0.5307 [N%8=0], GUE 0.5996 [N%8=2,6], GSE 0.6744 [N%8=4])\n";
    return 0;
}
