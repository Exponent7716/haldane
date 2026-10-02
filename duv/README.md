# duv — ArF液浸DUV露光装置（スキャナ）シミュレータ

193 nm ArF液浸スキャナの内部を、教科書・公開文献レベルの物理モデルで
サブシステムごとに再現した Python パッケージです。光源から現像後の線幅、
アライメント・重ね合わせ・スループットまでを一本のパイプラインで繋いでいます。

```
レーザー → 照明系 → レチクル → 投影レンズ + 水 → ウェハステージ → レジスト
                                    ↑                    ↑
                       計測側（アライメント / レベルセンサー）
```

## モジュール

| モジュール | 内容 |
|---|---|
| `core.py` | 単位系・定数（λ=193.368 nm, NA 1.35, n_water=1.4366, 4x）、`Grid` / `SourceMap` / `Mask` / `ImagingSettings` |
| `light_source.py` | ArFエキシマレーザー：放電レート方程式、F2劣化とガスインジェクション、狭帯域化モジュール（プリズム＋エシェル格子, E95）、波長安定化ループ、パルスごとのドーズ制御、スペックル |
| `illumination.py` | 照明形状（輪帯・ダイポール・クエーサー・自由形状）、マイクロミラーアレイ / DOE、偏光（TE/TM）、フライアイ均一化、スリット・REMAブレード、フィンガー式照度ムラ補正 |
| `reticle.py` | レイアウト（L/S・コンタクト・ラインエンド等）、バイナリ / 6% attPSM / altPSM、バイアス・ルールベースOPC・SRAF、境界層3Dマスク近似、ペリクル（Airy干渉）、MEEF、レチクルステージ、レチクル発熱 |
| `projection.py` | Fringe Zernike 1–37、液浸の厳密デフォーカス位相、Abbe結像（簡易ベクトル/偏光モデル）、フレア、色収差、レンズ収差指紋とマニピュレータ、レンズ発熱とフィードフォワード、液浸フード（水温→焦点） |
| `resist.py` | 膜スタック転送行列（BARC最適化・定在波）、化学増幅レジスト（Dill露光・PEB酸拡散・Mack現像）、CD測定、ドーズ・トゥ・サイズ、FEM / プロセスウィンドウ / Bossung、ショットノイズLER |
| `stage.py` | ジャーク制限スキャンプロファイル、サーボ（PID＋FF, 外乱, 柔軟モード）、MA/MSD、同期誤差、エンコーダ/干渉計、ショットレイアウト、デュアルステージとスループット |
| `metrology.py` | ウェハ変形モデル、多波長位相格子アライメントセンサー（マーク非対称補正）、線形・高次グリッドモデル、レベルセンサーとフォーカス補正、オーバーレイ統計・ショット内モデル、EWMA ラン・トゥ・ラン制御 |
| `scanner.py` | 全体統合 `DUVScanner`：1点露光、ドーズ・トゥ・サイズ、プロセスウィンドウ、ウェハ1枚分の露光（CDU・オーバーレイ・スループット） |

## 使い方

```bash
pip install numpy scipy matplotlib pytest
cd duv
python3 -m pytest -q                       # 全テスト
python3 examples/run_scanner.py --plot     # デモ（examples/output/ に図を出力）
```

```python
from duv import reticle
from duv.scanner import DUVScanner, ScannerConfig

s = DUVScanner(ScannerConfig(illumination="dipole_x", polarization="Y"))
g = s.config.grid
mask = reticle.make_mask(reticle.line_space(g, 45, 128), g, "attpsm")
dose = s.dose_to_size(mask, 45.0)          # mJ/cm^2
print(s.expose_field(mask, dose, defocus_nm=40).cd)
print(s.expose_wafer(mask, 45.0, n_fields=12)["overlay_stats"])
```

## 前提と限界

- 薄膜（Kirchhoff）マスク＋境界層近似で、厳密電磁界計算（RCWA/FDTD）ではありません。
- ベクトル結像は TE/TM 分解による簡易モデルで、レジスト内の膜による偏光効果やレンズの偏光収差は含みません。
- レンズ発熱の感度、外乱レベル、ウェハ形状などは現実的な桁になるよう選んだ例示値で、実機で校正した値ではありません。
- そのため `expose_wafer` の CDU やオーバーレイは、モデルに入っている誤差要因だけの合算です（マスクCD誤差、レジストプロセスのばらつき、エッチング等は含まれていません）。
- メーカー固有の設計値は使っておらず、公開文献に出てくる代表的な値・物理モデルだけで構成しています。
