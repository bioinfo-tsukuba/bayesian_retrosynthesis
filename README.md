# Bayesian Algorithm for Retrosynthesis

論文「Bayesian Algorithm for Retrosynthesis」の実装です。
論文URL: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00320

## 概要

このプロジェクトは、ベイジアンアプローチを用いた逆合成経路予測アルゴリズムを実装しています。モンテカルロサンプリングと条件付き確率分布P(S|Y)を使用して、目標分子から出発原料への最適な合成経路を予測します。

## 主要な特徴

- **ベイジアン推論**: ベイズの定理を用いた確率的逆合成予測
- **モンテカルロサンプリング**: 多様な合成経路の探索
- **分子トランスフォーマー**: 前進反応予測のための深層学習モデル
- **評価システム**: 論文と同様のベンチマーク評価
- **インタラクティブデモ**: 使いやすいデモンストレーション

## アルゴリズムの核心

### 1. ベイジアン逆合成
```
P(S|Y) = P(Y|S) * P(S) / P(Y)
```

ここで：
- P(S|Y): 目標分子Yが与えられたときの出発原料Sの確率
- P(Y|S): 前進反応確率（分子トランスフォーマーから）
- P(S): 出発原料の事前確率
- P(Y): 目標分子の事前確率

### 2. モンテカルロサンプリング
- 温度パラメータによる探索と活用のバランス
- 複数の合成経路の確率的生成
- 経路スコアリングによる最適化

## ファイル構成

```
bayesian_retrosynthesis/
├── bayesian_retrosynthesis.py    # メインアルゴリズム
├── molecular_transformer.py      # 分子トランスフォーマーモデル
├── evaluation.py                 # 評価・ベンチマークシステム
├── demo.py                       # インタラクティブデモ
├── requirements.txt              # 依存関係
└── README.md                     # このファイル
```

## インストール

### 必要な依存関係

```bash
pip install -r requirements.txt
```

### 主要な依存関係
- numpy: 数値計算
- matplotlib: グラフ描画
- seaborn: 統計的可視化

## 使用方法

### 1. 基本的な使用例

```python
from bayesian_retrosynthesis import BayesianRetrosynthesis, Molecule
from molecular_transformer import MolecularTransformer

# 分子トランスフォーマーを作成
transformer = MolecularTransformer()

# ベイジアン逆合成予測器を作成
predictor = BayesianRetrosynthesis(
    forward_model=transformer,
    max_depth=4,
    num_samples=1000,
    temperature=1.0
)

# 目標分子を定義
target = Molecule("CC(=O)OCC")  # 酢酸エチル

# 逆合成経路を予測
pathways = predictor.predict_retrosynthesis(target, num_pathways=5)

# 結果を表示
for i, pathway in enumerate(pathways, 1):
    print(f"経路 {i}:")
    for reaction in pathway:
        reactants = '.'.join([mol.smiles for mol in reaction.reactants])
        products = '.'.join([mol.smiles for mol in reaction.products])
        print(f"  {reactants} >> {products} (確率: {reaction.probability:.3f})")
```

### 2. インタラクティブデモの実行

```bash
cd bayesian_retrosynthesis
python demo.py
```

デモメニュー：
1. 単一分子の逆合成
2. 手法比較
3. インタラクティブモード
4. 評価システム
5. 分子トランスフォーマー
6. 全デモ実行

### 3. 評価の実行

```python
from evaluation import RetrosynthesisEvaluator

evaluator = RetrosynthesisEvaluator()

# 複数手法の比較
methods = [
    (bayesian_method, "Bayesian"),
    (baseline_method, "Baseline")
]

results = evaluator.compare_methods(methods)
```

### 4. 分子トランスフォーマーの単独使用

```python
from molecular_transformer import MolecularTransformer

transformer = MolecularTransformer()
reactants = [Molecule("CCO")]  # エタノール

predictions = transformer.predict(reactants)
for products, probability in predictions:
    product_smiles = '.'.join([mol.smiles for mol in products])
    print(f"{product_smiles} (確率: {probability:.3f})")
```

## アルゴリズムの詳細

### ベイジアン推論プロセス

1. **前進モデル**: 反応物から生成物を予測
2. **事前確率**: 分子の複雑さと入手可能性に基づく
3. **ベイズ更新**: 条件付き確率の計算
4. **モンテカルロサンプリング**: 確率的経路探索

### 分子表現

- **SMILES記法**: 分子の文字列表現
- **複雑度スコア**: 分岐、結合、特殊原子に基づく
- **分子量推定**: SMILES長に基づく近似

### 反応テンプレート

実装されている反応タイプ：
- 鈴木カップリング
- グリニャール反応
- アルドール縮合
- ディールス・アルダー反応
- 求核置換反応
- 酸化・還元反応
- エステル化反応

## 評価指標

論文と同様の評価指標を実装：

- **Top-k精度**: 上位k個の予測における正解率
- **実行時間**: 平均予測時間
- **有効経路率**: 化学的に妥当な経路の割合

## 実装の特徴

### 1. モジュラー設計
- 前進モデルの交換可能性
- 評価システムの独立性
- 拡張しやすい構造

### 2. 確率的アプローチ
- 温度パラメータによる制御
- 多様性と精度のバランス
- ロバストな予測

### 3. 実用的な機能
- インタラクティブデモ
- 詳細な評価レポート
- 可視化機能

## 論文との対応

この実装は以下の論文の概念を忠実に再現しています：

1. **ベイジアンフレームワーク**: P(S|Y)の計算
2. **モンテカルロサンプリング**: 経路探索手法
3. **評価指標**: Top-k精度の比較
4. **ベンチマーク**: 他手法との性能比較

## 制限事項

- 簡略化された分子表現（実際のRDKitなどは使用せず）
- 限定的な反応テンプレート
- 基本的な分子複雑度計算
- デモンストレーション目的の実装

## 今後の拡張可能性

1. **RDKit統合**: より正確な分子処理
2. **大規模データセット**: 実際の反応データベース
3. **深層学習モデル**: より高度な前進予測
4. **反応条件予測**: 温度、触媒、溶媒の考慮
5. **3D構造考慮**: 立体化学の取り扱い

## 参考文献

- Guo, Z., Wu, S., Ohno, M., & Yoshida, R. (2020). Bayesian Algorithm for Retrosynthesis. Journal of Chemical Information and Modeling, 60(9), 4474-4486.

## ライセンス

このプロジェクトは教育・研究目的で作成されています。

## 作成者

論文の実装再現プロジェクト

---

**注意**: この実装は論文の概念を理解し、デモンストレーションするためのものです。実際の化学合成計画には、より高度で検証されたツールの使用を推奨します。
