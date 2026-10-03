"""EUVリソグラフィ確率論的欠陥シミュレーション (コンタクトホール配列).

露光量 56 -> 70 -> 84 mJ/cm^2 で
  * 中央露出不足 (ホール中心が現像されない: missing / under-exposure)
  * パターン合体 (隣接ホールが繋がる: merging / bridging)
の確率がどう変わるかをモンテカルロ法で求め、画像化する。

モデル: 光学像(ガウスぼけ) -> 光子ショットノイズ(ポアソン) -> 二次電子/酸拡散ぼけ
        -> 閾値現像(ポジ型: 閾値以上が溶解) -> 連結成分解析で欠陥判定
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

E_PHOTON_EV = 91.8                 # 13.5 nm 光子エネルギー
EV_J = 1.602e-19
PH_PER_NM2_PER_MJCM2 = 1e-17 / (E_PHOTON_EV * EV_J)   # ≈0.68 photons/nm^2 per mJ/cm^2

# --- 条件 ---
PITCH = 36        # nm
CD = 18           # nm (マスク上のホール径)
NX = 7            # NX x NX のホール配列
MARGIN = 24
ABSORB = 0.03     # 実効吸収/酸発生効率
SIGMA_OPT = 6.0   # nm 光学ぼけ
SIGMA_RES = 1.5   # nm 二次電子+酸拡散ぼけ
FLARE = 0.22
THRESH = 0.55     # 現像閾値 (吸収光子 /nm^2 に対する比率, 基準線量 70 で規格化)
REF_DOSE = 70.0
CENTER_R = 4      # 中央判定半径 nm
DOSES = [56, 70, 84]


def hole_grid():
    size = NX * PITCH + 2 * MARGIN
    yy, xx = np.mgrid[0:size, 0:size]
    centers = [(MARGIN + PITCH * (i + .5), MARGIN + PITCH * (j + .5))
               for j in range(NX) for i in range(NX)]
    mask = np.zeros((size, size))
    for cx, cy in centers:
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= (CD / 2) ** 2] = 1
    return size, xx, yy, centers, mask


SIZE, XX, YY, CENTERS, MASK = hole_grid()
AERIAL = FLARE + (1 - FLARE) * gaussian_filter(MASK, SIGMA_OPT)
AERIAL /= AERIAL[tuple(int(c) for c in CENTERS[len(CENTERS) // 2][::-1])]  # 中心=1
CENTER_PIX = [((XX - cx) ** 2 + (YY - cy) ** 2 <= CENTER_R ** 2) for cx, cy in CENTERS]
INTERIOR = [k for k in range(len(CENTERS))
            if 0 < k % NX < NX - 1 and 0 < k // NX < NX - 1]   # 端効果を除く


def expose(dose, rng):
    mean_abs = dose * PH_PER_NM2_PER_MJCM2 * ABSORB * AERIAL
    photons = rng.poisson(mean_abs)
    acid = gaussian_filter(photons.astype(float), SIGMA_RES)
    ref = REF_DOSE * PH_PER_NM2_PER_MJCM2 * ABSORB
    return acid, acid > THRESH * ref


def classify(dev):
    """各内側ホールについて (missing, merged) を返す."""
    lab, _ = label(dev)
    missing, merged = {}, {}
    labels_at = [np.unique(lab[m & dev]) for m in CENTER_PIX]
    for k in INTERIOR:
        miss = not dev[CENTER_PIX[k]].all()
        missing[k] = miss
        if miss:
            merged[k] = False
            continue
        l = lab[CENTER_PIX[k]][0]
        merged[k] = any(l in labels_at[j] for j in range(len(CENTERS)) if j != k
                        and dev[CENTER_PIX[j]].all())
    return missing, merged


def run(n_trials, seed):
    rng = np.random.default_rng(seed)
    stats = {}
    for d in DOSES:
        pm = pg = 0
        for _ in range(n_trials):
            _, dev = expose(d, rng)
            m, g = classify(dev)
            pm += sum(m.values()); pg += sum(g.values())
        n = n_trials * len(INTERIOR)
        stats[d] = (pm / n, pg / n)
        print(f"dose {d:3d} mJ/cm2: missing {pm / n:7.3%}  merging {pg / n:7.3%}")
    return stats


def render(stats, seed, out):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.28, wspace=0.12)
    cmap = ListedColormap(["#1b1b1b", "#f2f2f2"])
    for c, d in enumerate(DOSES):
        # 欠陥を含む代表例を探す (見やすさのため先頭で見つかった試行を採用)
        for _ in range(200):
            acid, dev = expose(d, rng)
            m, g = classify(dev)
            if (any(m.values()) if c == 0 else any(g.values()) if c == 2 else True):
                break
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(acid, cmap="inferno", origin="lower")
        ax.set_title(f"{d} mJ/cm² ({d*PH_PER_NM2_PER_MJCM2:.0f} photons/nm²)\n光酸発生密度", fontsize=11)
        ax.axis("off")
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(dev, cmap=cmap, origin="lower")
        for k in INTERIOR:
            cx, cy = CENTERS[k]
            if m[k]:
                ax.add_patch(plt.Circle((cx, cy), CD * .8, fill=False, ec="#2a9df4", lw=2))
            elif g[k]:
                ax.add_patch(plt.Circle((cx, cy), CD * .8, fill=False, ec="#ff3b30", lw=2))
        ax.set_title(f"現像後 (白=開口)\n青:中央露出不足 {sum(m.values())} / 赤:合体 {sum(g.values())}", fontsize=11)
        ax.axis("off")
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(len(DOSES)); w = 0.35
    pm = [stats[d][0] * 100 for d in DOSES]; pg = [stats[d][1] * 100 for d in DOSES]
    b1 = ax.bar(x - w / 2, pm, w, color="#2a9df4", label="中央露出不足 (missing)")
    b2 = ax.bar(x + w / 2, pg, w, color="#ff3b30", label="パターン合体 (merging)")
    ax.bar_label(b1, fmt="%.1f%%"); ax.bar_label(b2, fmt="%.1f%%")
    ax.set_xticks(x, [f"{d} mJ/cm²" for d in DOSES]); ax.set_ylabel("ホール当たり確率 [%]")
    ax.legend(); ax.set_title("露光量依存性 (モンテカルロ)")
    fig.suptitle("EUVリソグラフィ 確率論的欠陥シミュレーション: 露光量 56→70→84 mJ/cm²", fontsize=14)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="euv/euv_dose_simulation.png")
    a = ap.parse_args()
    plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]
    render(run(a.trials, a.seed), a.seed, a.out)
