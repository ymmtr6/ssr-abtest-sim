# Method

`ssr-abtest-sim` screens web-page variants by the Semantic Similarity Rating
(SSR) method in Maier et al., *LLMs Reproduce Human Purchase Intent via
Semantic Similarity Elicitation of Likert Ratings*, arXiv:2510.08338v3
(27 October 2025), <https://arxiv.org/abs/2510.08338>.

Each persona sees one variant (between subjects). The model writes a short
free-text reaction. The reaction is not a number. Embeddings of that
reaction are compared with embeddings of five anchor statements, and the
same comparison is repeated for every anchor set. The sets are averaged.

Scaled cosine, matching the paper's reference implementation:

```text
γ(u, v) = (1 + cosine(u, v)) / 2
```

Probability of Likert point `r`:

```text
p(r) = (γ_r − γ_ℓ + ε δ_{ℓ,r}) / (Σ_s (γ_s − γ_ℓ) + ε)
```

`γ_ℓ` is the smallest similarity in the row and `ℓ` is its first index.
The paper's reported runs use `ε = 0` and temperature `T = 1`. At `ε = 0`
the probabilities are invariant to a positive affine transform of `γ`, so
raw cosine and scaled cosine agree. A positive `ε` is applied in scaled
cosine space.

If every anchor is equally similar and `ε = 0`, the ratio is undefined.
This package returns the uniform distribution, which is the limit of the
ratio as the similarities approach each other.

Temperature scaling, after the sets have been averaged, is
`p_r ← p_r^(1/T) / Z`. `T = 1` leaves the distribution unchanged. `T = 0`
puts all mass on the first mode. A uniform row stays uniform. Applying
temperature after the average, rather than inside each set, matches the
paper's reference rater.

The primary number in the report is the expected rating on a 1–5 scale,
averaged across anchor sets and across a persona's samples. Arms are
compared with a bootstrap confidence interval for the difference of means,
a two-sample Kolmogorov-Smirnov statistic on the persona-level expected
ratings, top-two-box probabilities, and Cohen's d.

The optional Decisions API call (`POST /v1/decisions`, model `gpt-6-luna`)
scores the same free text. It is an auxiliary check. It is not SSR.

`input_mode: image` or `both` attaches a screenshot to the Responses
request as an `input_image` part next to the question. `text` does not.
The image is a local HTML capture, a URL capture, or a file you already
have. Token estimates for that image use the vision guide's patch or
tile rules.

Bundled anchor sentences were written for this tool. They are not the
personal-care purchase-intent anchors from the paper.
