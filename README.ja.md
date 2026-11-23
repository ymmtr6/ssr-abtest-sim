# ssr-abtest-sim

LLM のペルソナと [Semantic Similarity Rating](https://arxiv.org/abs/2510.08338)（SSR）で、ウェブページの案を比較するツールです。

ペルソナはランディングページ、商品ページ、その他のページ案のうち一つだけを読み、短い自由記述で反応します。その文章の埋め込みを、5段階のアンカー文との類似度で 1〜5 の分布に変換します。レポートが比べるのは、関連性やクリック意向のような**表明された態度**です。クリック率、コンバージョン率、滞在時間の推定ではなく、実地の A/B テストの代わりでもありません。

例は架空の週末旅行のランディングページです。API キーなしで流れを確認できます。

```bash
pip install -e ".[dev]"
ssr-abtest run configs/experiments/example_lp.yaml --dry-run
```

`outputs/lp-hero-copy/report.json` と `report.md` を書きます。ドライランの数値はローカルの固定データで、モデルの判断ではありません。

## 引用

Maier ほか, *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*, arXiv:2510.08338v3（2025年10月27日）, <https://arxiv.org/abs/2510.08338>.

確率への変換は、この論文と Apache-2.0 の参照実装（<https://github.com/pymc-labs/semantic-similarity-rating>）に沿っています。参照実装のコードは同梱していません。同梱のアンカー文はページ評価用に新たに書いたもので、論文のパーソナルケア商品の購入意向アンカーではありません。

<https://arxiv.org/bibtex/2510.08338> から取得した BibTeX:

```bibtex
@misc{maier2025llmsreproducehumanpurchase,
      title={LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings}, 
      author={Benjamin F. Maier and Ulf Aslak and Luca Fiaschi and Nina Rismal and Kemble Fletcher and Christian C. Luhmann and Robbie Dow and Kli Pappas and Thomas V. Wiecki},
      year={2025},
      eprint={2510.08338},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.08338}, 
}
```

## インストール

Python 3.11 以降。

```bash
pip install -e .
```

スクリーンショットは任意です。例の実行には不要です。

```bash
pip install -e ".[screenshot]"
playwright install chromium
```

本番実行のときだけキーを渡します。SDK は環境変数 `OPENAI_API_KEY` を読みます。リポジトリには置かないでください。

```bash
export OPENAI_API_KEY="..."
ssr-abtest run configs/experiments/example_lp.yaml --output outputs
```

本番実行は Responses、Embeddings、Decisions の応答を `.ssr-abtest-cache/`（gitignore 済み）に保存します。再実行では保存済みのトークン数をレポートに戻し、同じ呼び出しは送りません。`--no-cache` で無効にできます。

`ssr-abtest estimate configs/experiments/example_lp.yaml` は費用の見積もりだけを出し、API は呼びません。単価は 2026-11-23 に確認した標準料金で、Batch、Flex、Fast、キャッシュ割引、リージョン処理の割増は含みません。詳細は `src/ssr_abtest/cost.py` です。

## 処理の流れ

1. 属性ごとの周辺分布から独立にペルソナを引き、各人を一つの案に割り当てます。割り当てはシード付きの均衡シャッフルです。同時分布の再現ではありません。
2. HTML またはテキストから見える文字列を取り出します。`render` に `screenshot` を含めると、任意依存関係があるときローカルに PNG を保存します。この版では画像をモデルに送りません。
3. Responses API（`client.responses.create`）で自由記述を生成します。プロンプトは数値の評定を禁じています。既定の生成モデルは `gpt-5.4-mini`、1人あたり 2 サンプル、温度 0.5 です。
4. 回答とアンカー文を `text-embedding-3-small` で埋め込みます。
5. スケールしたコサイン類似度を 5 段階の PMF にします（既定は ε = 0、温度 T = 1）。アンカーセットは等加重で平均します。
6. ペルソナ単位の平均リッカート得点について、ブートストラップ信頼区間、二元標本の Kolmogorov-Smirnov 統計量、Top-2 box、Cohen's d を出します。
7. JSON と Markdown を書きます。性別と地域の内訳は探索的です。

`--decisions` を付けると、各自由記述に Decisions API（`POST /v1/decisions`、モデル `gpt-6-luna`）を追加で呼びます。2026-11-23 の Decisions ガイドに合わせて確認しています。送るのは 5 段階の `score` が 1 問と、`predicate` が 2 問（主要なボタンを押したそうか、ペルソナから外れているか）です。`score` の `value` は 0 始まりの添字で、レポートでは 1〜5 の平均に直します。ペルソナ逸脱の確率が `persona_break_threshold`（既定 0.8）以上の回答は集計から除きます。API には `choice` 型もありますが、既定の質問セットには入れていません。主指標は SSR のままです。Decisions の得点は補助で、論文で直接評定が中央に寄ったのと同じ性質を持ち得ます。

## 配置

```text
src/ssr_abtest/      ライブラリと ssr-abtest コマンド
configs/personas/    周辺分布の重み（集計値のみ）
configs/anchors/     日本語と英語のアンカー文
configs/experiments  例の実験
examples/pages/      架空の LP、商品ページ、ルート一覧
tests/               プロバイダはモック。ネットワークは使わない
```

実験ファイル内のパスは、まずカレントディレクトリ、次にそのファイルの場所から探します。コマンドはリポジトリのルートで実行してください。

`configs/personas/jp_census_example.yaml` という名前は、周辺分布の表という形の例だからです。重みは丸めたプレースホルダで、公式統計ではありません。再配布してよい公開の集計表に差し替えてください。個票はコミットしないでください。

`examples/pages/` はすべて架空です。例が使う LP の対、商品ページ、そしてその架空の旅行サイト内のルート一覧（結果一覧のようなレイアウト）です。

## 設定

型は `configs/experiments/example_lp.yaml` です。各案は `path` か本文 `text` の一方です。質問はアンカーファイルを指し、各セットはちょうど 5 文で、先頭が段階 1 です。論文は参照セットを 6 種類平均しています。例は各質問 3 セットです。同じ形で増やせます。

英語のアンカーを使うときは、`language` を `en` にし、`scenario` と `questions` を英語にして、`anchors` を `configs/anchors/en/relevance.yaml` と `configs/anchors/en/click_intent.yaml` にします。

価格表にないモデルは、`models.prices.<model>.input` と `.output`（100万トークンあたり米ドル）を書かないと、ドル合計は空になります。

## テスト

```bash
ruff check src tests
pytest
```

GitHub Actions は Python 3.11 と 3.12 でこの二つを実行し、`OPENAI_API_KEY` を空にして例をドライランします。

## 限界

詳しくは [docs/limitations.md](docs/limitations.md) と [docs/method.md](docs/method.md) です。短く言うと、出るのは表明された態度だけで、アンカーは未校正で、人口属性の内訳は探索的です。公開アーカイブでの校正は今後の作業です（[benchmarks/upworthy/README.md](benchmarks/upworthy/README.md)）。

そのほかの続き: スクリーンショットをモデルに渡すこと、Batch API の料金、属性の同時分布、外部の勝敗データへの ε と T の適合。

## ライセンス

Apache-2.0。Copyright 2026 Riku。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。

English documentation: [README.md](README.md).
