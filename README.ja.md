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

例のドライランはデスクトップのスクリーンショット分のトークンを見積もりますが、ブラウザは起動しません。HTML や URL のスクリーンショットを本番で送るときは任意依存関係が必要です。

```bash
pip install -e ".[screenshot]"
playwright install chromium
```

本番実行のときだけキーを渡します。既定は `OPENAI_API_KEY` です。役割ごとに `ZAI_API_KEY` や `CF_AIG_TOKEN` も使えます。キーはリポジトリに置かないでください。z.ai は従量課金の標準キーを使います。GLM Coding Plan のキーは公式のコーディングツール向けで、ここには向きません。

```bash
export OPENAI_API_KEY="..."
ssr-abtest run configs/experiments/example_lp.yaml --output outputs
```

本番実行は Responses、Embeddings、Decisions の応答を `.ssr-abtest-cache/`（gitignore 済み）に保存します。再実行では保存済みのトークン数をレポートに戻し、同じ呼び出しは送りません。`--no-cache` で無効にできます。

`ssr-abtest estimate configs/experiments/example_lp.yaml` は費用の見積もりだけを出し、API は呼びません。単価は 2026-11-23 に確認した標準料金で、Batch、Flex、Fast、キャッシュ割引、リージョン処理の割増は含みません。詳細は `src/ssr_abtest/cost.py` です。

## 処理の流れ

1. 属性ごとの周辺分布から独立にペルソナを引き、各人を一つの案に割り当てます。割り当てはシード付きの均衡シャッフルです。同時分布の再現ではありません。
2. 各案はローカルの HTML かテキスト、`http(s)` の URL、または既存の画像ファイルです。`input_mode` は `text`、`image`、`both` です。`image` と `both` はスクリーンショットを添付します。HTML と URL の撮影は Playwright で、ビューポートはデスクトップ（1280×800）かモバイル（390×844）、範囲はファーストビューかページ全体です。本番の撮影結果は `.ssr-abtest-cache/screenshots/` にキャッシュします。ドライランはプレースホルダの PNG を書き、ブラウザの代わりに予定したピクセル数で費用を見積もります。
3. Responses API（`client.responses.create`）で自由記述を生成します。画像は質問の横に `input_image` として渡し、`detail` は設定値（既定 `high`）です。プロンプトは数値の評定を禁じています。既定の生成モデルは `gpt-5.4-mini`、1人あたり 2 サンプル、温度 0.5 です。画像トークンは 2026-11-23 に確認した vision ガイドに従って入力見積もりへ足します。現行モデルは 32×32 パッチにモデル倍率を掛けます（`gpt-5.4-mini` は 1.2）。古い vision モデルは基本トークンに 512px タイルを足します。`gpt-5.4-mini` では `detail: low` の方が `high` より高くなることがあります。本番実行の請求は API が返す usage です。
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

型は `configs/experiments/example_lp.yaml` です。各案は `path`（ローカルの HTML かテキスト）、`text`、`url`、`image` のどれか一つです。例は `input_mode: both` で、抽出した本文に加えてデスクトップのファーストビュー相当のスクリーンショットを添付します。質問はアンカーファイルを指し、各セットはちょうど 5 文で、先頭が段階 1 です。論文は参照セットを 6 種類平均しています。例は各質問 3 セットです。同じ形で増やせます。

英語のアンカーを使うときは、`language` を `en` にし、`scenario` と `questions` を英語にして、`anchors` を `configs/anchors/en/relevance.yaml` と `configs/anchors/en/click_intent.yaml` にします。

価格表にないモデルは、`models.prices.<model>.input` と `.output`（100万トークンあたり米ドル）を書かないと、ドル合計は空になります。

## プロバイダ

生成、埋め込み、Decisions は別々のプロバイダにできます。既定は OpenAI 直接で、文章は Responses、SSR の埋め込みは Embeddings、Decisions は `--decisions` のときだけです。`models.providers` に `configs/providers/` のファイルか、同じ形のマッピングを指定します。

- `openai.yaml` が既定です。
- `zai.yaml` は z.ai の Chat Completions（`https://api.z.ai/api/paas/v4/chat/completions`）で生成し、埋め込みは OpenAI のままです。`embedding-3` が文書化されているのは中国向け `https://open.bigmodel.cn/api/paas/v4/embeddings` だけで、バッチ 64、次元 2048、100万トークンあたり 0.5 人民元です。人民元は米ドル合計に足しません。
- `cloudflare-gateway.yaml` は OpenAI を `https://gateway.ai.cloudflare.com/v1/${CF_ACCOUNT_ID}/${CF_GATEWAY_ID}/openai` に通します。Responses と Chat Completions は文書化されています。Embeddings は同じベース URL の置き換えで、専用の公式例はありません。
- `cloudflare-zai.yaml` はカスタムプロバイダ経由の z.ai です。スラッグ `zai`、ベース URL `https://api.z.ai` をゲートウェイに作ります。呼び出しは `.../custom-zai/api/paas/v4/chat/completions` です。埋め込みは Workers AI の `@cf/baai/bge-m3` です。
- `cloudflare-workers.yaml` は Cloudflare 上の `@cf/zai-org/glm-5.3` と `@cf/baai/bge-m3` です。Workers AI の `glm-5.3-flash` は 2026-11-20 に非推奨になったので使いません。bge-m3 は `/ai/run/@cf/baai/bge-m3` に `{"text": [...]}` と `cf-aig-gateway-id` を送ります。

認証付きゲートウェイは `cf-aig-authorization: Bearer $CF_AIG_TOKEN` です。BYOK では `Authorization` を付けず、保存されたプロバイダ鍵を使います。生成は `cf-aig-skip-cache: true` を送ります。ゲートウェイのキャッシュ鍵はリクエスト本文全体を含むため、同じ本文のペルソナ再サンプルが衝突します。`cf-aig-cache-key` は送りません。埋め込みはゲートウェイのキャッシュを使って構いません。Decisions は z.ai には無く、ゲートウェイでも文書化されていません。decide を OpenAI 直接にするか、Decisions を切ってください。

埋め込みモデルを変えると SSR の尺度が変わるので、プロバイダごとに検証をやり直します。価格表にないモデルは 0 円とは扱わず、見積もりを空にして報告します。2026-11-23 に確認した国際価格（100万トークンあたり米ドル）は、GLM-5.3 が 1.4/4.4、GLM-5.3-Flash が 0.15/0.50、GLM-5.3-FlashX が 0.37/1.25、GLM-4.6V が 0.30/0.90、GLM-4.6V-FlashX が 0.04/0.40 です。

## 検証

`ssr-abtest validate` は Upworthy Research Archive の見出しテストを、週で層化したシード付き標本として採り、表明されたクリック意向を観測されたクリック率と比べます。アーカイブは CC BY 4.0 です。引用は Matias, Munger, Aubin Le Quere, Ebersole, Scientific Data (2021), <https://doi.org/10.1038/s41597-021-00934-7>、および OSF <https://doi.org/10.17605/OSF.IO/JD64P> です。取得とチェックサムは [benchmarks/upworthy/README.md](benchmarks/upworthy/README.md) にあります。CSV はこのリポジトリに含まれません。

レポートには、二群の比率の z 検定で分かれたペアだけについての勝敗一致率とブートストラップ区間、テスト内の Spearman 相関の平均、校正用のビン、ランダムなベースライン、見出しが長い方を選ぶベースラインが入ります。SSR と同時に 1〜5 の直接評定も計算します。`--decisions` で Decisions の得点を足します。`--dry-run` は費用を表示し、ローカルの固定データでスコアを書きます。本番実行には `--budget`（米ドル）が必要で、見積もりが上限を超えると API を呼びません。

```bash
python benchmarks/upworthy/fetch_archive.py --output data/upworthy
ssr-abtest validate data/upworthy/upworthy-archive-exploratory-packages-03.12.2020.csv --dry-run --n 20
```

## テスト

```bash
ruff check src tests
pytest
```

GitHub Actions は Python 3.11 と 3.12 でこの二つを実行し、`OPENAI_API_KEY` を空にして例をドライランします。

## 限界

詳しくは [docs/limitations.md](docs/limitations.md) と [docs/method.md](docs/method.md) です。短く言うと、出るのは表明された態度だけで、アンカーは未校正で、人口属性の内訳は探索的です。公開アーカイブでの校正は今後の作業です（[benchmarks/upworthy/README.md](benchmarks/upworthy/README.md)）。

そのほかの続き: Batch API の料金、属性の同時分布、外部の勝敗データへの ε と T の適合。

## ライセンス

Apache-2.0。Copyright 2026 Riku。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。

English documentation: [README.md](README.md).
