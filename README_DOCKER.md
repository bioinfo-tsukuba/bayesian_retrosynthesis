# Docker環境での実行方法

このプロジェクトをDocker環境で実行するための手順です。

## 前提条件

- Docker Desktop がインストールされていること
- Docker Compose がインストールされていること

## 基本的な使用方法

### 1. Dockerイメージのビルド

```bash
docker build -t bayesian-retrosynthesis .
```

### 2. インタラクティブデモの実行

```bash
# Docker Composeを使用（推奨）
docker compose up

# または直接Dockerコマンドを使用
docker run -it --rm bayesian-retrosynthesis
```

### 3. 開発環境での実行

開発用のシェル環境にアクセスする場合：

```bash
docker compose --profile dev up retrosynthesis-dev
```

コンテナ内でシェルを起動：

```bash
docker compose --profile dev run --rm retrosynthesis-dev /bin/bash
```

## 各スクリプトの実行方法

### メインアルゴリズムの実行

```bash
# Docker Compose使用
docker compose --profile main up retrosynthesis-main

# 直接実行
docker run --rm bayesian-retrosynthesis python bayesian_retrosynthesis.py
```

### 評価システムの実行

```bash
# Docker Compose使用
docker compose --profile eval up retrosynthesis-eval

# 直接実行
docker run --rm bayesian-retrosynthesis python evaluation.py
```

### 分子トランスフォーマーの実行

```bash
docker run --rm bayesian-retrosynthesis python molecular_transformer.py
```

### インタラクティブデモの実行

```bash
# デフォルトのサービス
docker compose up

# または
docker run -it --rm bayesian-retrosynthesis python demo.py
```

## ファイルの編集と開発

### ローカルファイルの変更を反映

docker-compose.ymlでは、ローカルディレクトリをコンテナにマウントしているため、ローカルでファイルを編集すると、コンテナ内でも即座に反映されます。

```bash
# ファイルを編集後、コンテナを再起動
docker compose restart
```

### 新しい依存関係の追加

requirements.txtを更新した場合は、イメージを再ビルドする必要があります：

```bash
docker compose build
```

## トラブルシューティング

### コンテナが起動しない場合

1. Dockerデーモンが起動していることを確認
2. ポート8000が他のプロセスで使用されていないか確認
3. イメージを再ビルド：

```bash
docker compose build --no-cache
```

### 権限エラーが発生する場合

コンテナ内でファイルの権限エラーが発生する場合：

```bash
# ローカルファイルの権限を確認
ls -la

# 必要に応じて権限を変更
chmod -R 755 .
```

### メモリ不足エラー

大きなサンプル数で実行する場合、メモリ不足が発生する可能性があります：

```bash
# Dockerのメモリ制限を増やす
docker run --memory=4g -it --rm bayesian-retrosynthesis
```

## 高度な使用方法

### カスタムパラメータでの実行

```bash
# 環境変数を使用してパラメータを設定
docker run --rm \
  -e MAX_DEPTH=5 \
  -e NUM_SAMPLES=500 \
  -e TEMPERATURE=1.5 \
  bayesian-retrosynthesis python bayesian_retrosynthesis.py
```

### 結果ファイルの保存

```bash
# 結果を保存するためのボリュームマウント
docker run --rm \
  -v $(pwd)/results:/app/results \
  bayesian-retrosynthesis python evaluation.py
```

### 複数のコンテナでの並列実行

```bash
# 複数のターゲット分子を並列処理
docker compose up --scale bayesian-retrosynthesis=3
```

## Docker環境の詳細

### 使用しているベースイメージ
- `python:3.9-slim`: 軽量なPython 3.9環境

### インストールされるパッケージ
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

### セキュリティ設定
- 非rootユーザー（retrosynthesis）での実行
- 最小限のシステムパッケージのみインストール

### ポート設定
- 8000番ポート：将来的なWeb UIのために予約

## パフォーマンス最適化

### マルチステージビルドの使用

より効率的なイメージサイズのために、Dockerfileを以下のように変更することも可能です：

```dockerfile
# マルチステージビルドの例
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "demo.py"]
```

### キャッシュの活用

```bash
# ビルドキャッシュを活用した高速ビルド
docker build --cache-from bayesian-retrosynthesis .
```

## 本番環境での使用

### セキュリティ強化

```bash
# セキュリティスキャン
docker scan bayesian-retrosynthesis

# 読み取り専用ファイルシステムでの実行
docker run --read-only --tmpfs /tmp --rm bayesian-retrosynthesis
```

### リソース制限

```bash
# CPU・メモリ制限付きで実行
docker run --cpus="2" --memory="2g" --rm bayesian-retrosynthesis
```

## 参考情報

- [Docker公式ドキュメント](https://docs.docker.com/)
- [Docker Compose公式ドキュメント](https://docs.docker.com/compose/)
- [Python Docker イメージ](https://hub.docker.com/_/python)

## サポート

Docker環境での問題が発生した場合は、以下を確認してください：

1. Docker Desktop のバージョン
2. 利用可能なメモリとディスク容量
3. ネットワーク接続
4. ファイアウォール設定

詳細なログは以下のコマンドで確認できます：

```bash
docker compose logs
