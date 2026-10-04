# euvsim — EUV露光装置 全サブシステム物理シミュレータ

EUVリソグラフィスキャナ（NA 0.33 の NXE:3x00 クラス、NA 0.55 アナモルフィックの EXE:5000 クラス）を、
公開されている物理・文献値だけで光源からウェハまで再現する教育・検討用シミュレータです。
特定メーカーの設計データの再現ではありません。

```
CO2レーザー ─► Sn液滴 ─► LPPプラズマ(13.5nm) ─► コレクタ ─► IF
 ─► 照明系(フィールド/瞳ファセットミラー) ─► レチクル(反射型マスク, ペリクル)
 ─► 投影光学系(6枚/8枚ミラー) ─► ウェハ(レジスト)          ※全経路 H2 真空中
```

## サブシステム

| パッケージ | 内容 |
|---|---|
| `euvsim/optics` | Mo/Si 多層膜ミラー反射率（Parratt法、ラフネス・相互拡散・Ru キャップ） |
| `euvsim/source` | CO2ドライブレーザー（プリパルス＋メインパルス）、Sn液滴生成、プラズマ変換効率・スペクトル、デブリ/H2緩和、楕円コレクタ、ドーズ制御 |
| `euvsim/illumination` | 瞳形状（通常/輪帯/ダイポール/四重極/フリーフォーム）、FlexPupil、フライアイ積分器、円弧スリット、UNICOM 均一性補正、透過率、エタンデュ |
| `euvsim/imaging` | パターン生成、反射型マスク（TaBN/Ni/attPSM 吸収体、シャドウイング＝M3D）、ペリクル、投影光学系（Zernike収差・フレア・中心遮蔽・アナモルフィック）、Abbe結像、NILS・プロセスウィンドウ |
| `euvsim/resist` | 光子ショットノイズ、CAR/MOR、二次電子ブラー、PEB酸拡散・クエンチャ、現像（閾値/Mack）、CD・LER/LWR・LCDU・確率的欠陥 |
| `euvsim/stage` | ジャーク/スナップ制限軌道、マグレブステージ制御とMA/MSD同期誤差、干渉計/エンコーダ、アライメント、レベルセンサ、オーバーレイ解析、ドーズ制御、デュアルステージのスループット |
| `euvsim/environment` | 真空・H2パージ、ダイナミックガスロック、EUVガス吸収、炭素汚染/酸化/H*クリーニング、ミラー・レチクル・ペリクル加熱 |
| `euvsim/scanner.py` | 全体統合：光量収支、スループット(WPH)、パターン露光→CD/LER |

## 使い方

```bash
pip install -r requirements.txt
python3 -m euvsim                 # NA 0.33, 32nm pitch L/S
python3 -m euvsim --high-na       # NA 0.55, 20nm pitch L/S
python3 -m euvsim --dose 60 --pitch 36
python3 -m pytest tests -q        # 全テスト
```

```python
from euvsim.scanner import EUVScanner
s = EUVScanner()
print(s.report(dose_mj_cm2=30))
r = s.print_lines(pitch_nm=32, cd_nm=16)
print(r["cd_nm"], r["ler_3sigma_nm"])
```

## 既定設定での主な結果（NA 0.33）

| 項目 | 値 |
|---|---|
| IFでのEUV出力 | 268 W（CE ≈ 5.3 %, CO2 22 kW, 50 kHz） |
| 照明系透過率 / レチクル入射 | 0.225 / 60 W |
| マスクブランク反射率 / 投影系透過率(6枚) | 0.72 / 0.16 |
| ウェハ到達パワー | 6.8 W（IF の約 2.5 %） |
| スループット @30 mJ/cm² | 約 207 枚/時（ステージ速度律速） |
| 32 nm pitch L/S | NILS 3.1, ドーズ 51 mJ/cm², LER 1.6 nm(3σ) |
| オーバーレイ（マッチドマシン） | 約 1.8 nm |

## 主な単純化・限界

- 結像はスカラー（高NAの偏光効果なし）。M3D は薄膜マスク＋シャドウ帯の近似で、ベストフォーカスシフトは厳密計算より小さめ。
- レジストの LER は実測(3–4 nm)より小さめ（PAG・脱保護の分子ノイズ未考慮）。
- 光源の一部損失係数（ビーム伝送、遮蔽など）は文献相当値を仮定値として置いている。
- 投影系ミラーは理想反射率を使うため、ウェハ到達パワーはやや楽観的。

モジュール間の取り決めは [ARCHITECTURE.md](ARCHITECTURE.md) を参照。
