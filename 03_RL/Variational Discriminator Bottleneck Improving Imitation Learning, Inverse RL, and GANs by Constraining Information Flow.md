# Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow

## Abstruct

敵対的学習（Adversarial Learning）手法は幅広い応用分野で提案されているが、敵対的モデルの学習は非常に不安定になりやすい。特に、生成器（Generator）と識別器（Discriminator）の性能バランスを適切に取ることが重要である。識別器の精度が極端に高くなると、得られる勾配がほとんど情報を持たなくなり、学習が困難になるためである。本研究では、識別器の情報の流れを制約するためのシンプルで汎用的な手法を提案する。具体的には、情報ボトルネック（Information Bottleneck） を導入し、観測データと識別器の内部表現との間の相互情報量（Mutual Information）を制約する。これにより、識別器の精度を適切に調整し、有益な勾配情報を維持できる。本研究では、提案手法である変分識別器ボトルネック（Variational Discriminator Bottleneck, VDB） が、敵対的学習アルゴリズムの3つの異なる応用分野において顕著な性能向上をもたらすことを示す。

1. 模倣学習（Imitation Learning）への応用
    - VDBを動的な連続制御スキル（例：走る動作）の模倣学習に適用。
    - 生のビデオデモンストレーション（Raw Video Demonstrations）から直接学習可能であり、従来の敵対的模倣学習手法を大幅に上回る性能を達成。
1. 逆強化学習（Adversarial Inverse Reinforcement Learning）との統合
    - VDBを逆強化学習と組み合わせ、シンプルで転移可能な報酬関数を学習可能にする。
    - 学習した報酬関数は新しい環境へ転移し、再最適化が可能。
1. 敵対的生成ネットワーク（GAN）での画像生成
    - VDBをGANの識別器に導入することで、学習の安定性を向上し、既存の安定化手法を上回る性能を実現。

本研究は、識別器の情報流を制御することで、敵対的学習の安定性と性能を向上させる新たな手法を提供する。

## Introduction
敵対的学習（Adversarial Learning）手法は、高次元データの複雑な内部相関構造を持つ分布をモデル化するための有望なアプローチを提供する。これらの手法では通常、識別器（Discriminator）を使用して生成器（Generator）の学習を監督し、元のデータと区別がつかないサンプルを生成できるようにする。その代表的な実装例として敵対的生成ネットワーク（Generative Adversarial Networks, GAN）があり、これを用いることで高精度な画像生成（Goodfellow et al., 2014; Karras et al., 2017）や、その他の高次元データの生成（Vondrick et al., 2016; Xie et al., 2018; Donahue et al., 2018）が可能となる。また、敵対的手法は逆強化学習（Inverse Reinforcement Learning, IRL）における報酬関数の学習（Finn et al., 2016a; Fu et al., 2017）や、デモンストレーションの直接模倣（Ho & Ermon, 2016）にも応用されている。
しかし、これらの手法は最適化に関する重大な課題を抱えており、その一つが生成器と識別器の性能バランスの維持である。

- 識別器の精度が高すぎると、生成器への勾配情報が乏しくなり、学習が困難になる。
- 逆に識別器が弱すぎると、生成器が適切に学習できない。

これらの課題により、敵対的学習アルゴリズムの安定化手法への関心が広がり、多くの研究が行われている（Arjovsky et al., 2017; Kodali et al., 2017; Berthelot et al., 2017）

本研究では、敵対的学習（Adversarial Learning）に対するシンプルな正則化手法を提案する。本手法では、情報ボトルネック（Information Bottleneck）の変分近似を用いて、入力から識別器（Discriminator）への情報の流れを制約する。

具体的には、入力観測データと識別器の内部表現との間の相互情報量（Mutual Information）を制約することで、識別器がデータ分布と生成器の分布の間に大きな重なりを持つ表現を学習するように促す。これにより、識別器の精度を適切に調整し、生成器（Generator）に対して有益かつ情報量の多い勾配を維持することができる。

本手法による敵対的学習の安定化は、インスタンスノイズ（Instance Noise）の適応的な変種（Salimans et al., 2016; Sønderby et al., 2016; Arjovsky & Bottou, 2017）とみなすことができる。しかし、本研究では、この手法の適応的な特性が極めて重要であることを示す。

- 識別器の内部表現と入力との間の相互情報量を制約することで、識別器の精度を直接制御できる。
- これにより、ノイズの大きさを自動で決定し、生成器とデータ分布の最も識別的な差異を学習するために最適化された圧縮表現にノイズを適用することが可能となる。

本研究の主な貢献は、変分識別器ボトルネック（Variational Discriminator Bottleneck, VDB）である。これは、敵対的学習（Adversarial Learning）における適応的な確率的正則化手法であり、さまざまな応用領域において性能を大幅に向上させる。その具体的な例は図1に示されている。

本手法は、多様なタスクやアーキテクチャに容易に適用可能である。

- 模倣学習（Imitation Learning）への適用
    - 本手法を、モーションキャプチャ（Mocap）データを用いたシミュレーションヒューマノイドの高度なアクロバティックスキルの学習など、難易度の高い模倣タスクの評価に用いる。
    - また、本手法は、生のビデオデモンストレーション（Raw Video Demonstrations）から直接動的な連続制御スキルを学習することを可能にし、従来の敵対的模倣学習手法を大幅に上回る性能を達成する。

- 逆強化学習（Inverse Reinforcement Learning）への適用
    - 本手法を用いることで、デモンストレーションから報酬関数を回復し、将来のポリシーを学習することが可能となる。

- 画像生成（Image Generation）への適用
    - 本手法を敵対的生成ネットワーク（GAN）に適用することで、多くのケースにおいて生成性能を向上させる。
本研究は、敵対的学習のさまざまな応用に対して、VDBが有効に機能することを示し、その適用範囲の広さと性能向上の効果を実証する。

## Related work
近年、敵対的学習（Adversarial Learning）手法が急速に発展しており、その背景には敵対的生成ネットワーク（GANs）の成功がある（Goodfellow et al., 2014）。GAN のフレームワークは、一般的に 識別器（Discriminator）と生成器（Generator）から構成される。識別器はサンプルが本物か偽物かを分類することを目的とし、生成器は識別器を欺くサンプルを生成することを目的とする。

逆強化学習（IRL）（Finn et al., 2016b）や 模倣学習（Imitation Learning）（Ho & Ermon, 2016）においても、類似のフレームワークが提案されている。

しかし、敵対的モデルの学習は非常に不安定になりやすく、特に識別器と生成器のバランスを取ることが大きな課題となっている（Berthelot et al., 2017）。識別器の性能が過剰に高くなると、生成器が容易に識別され、学習に有益な勾配情報を得られなくなる（Che et al., 2016）。

この問題を軽減するために、さまざまな代替損失関数（Alternative Loss Functions）が提案されている（Mao et al., 2016; Zhao et al., 2016; Arjovsky et al., 2017）。また、学習の安定性と収束性を向上させるため、以下のような正則化手法（Regularization Methods）も導入されている。

- 勾配ペナルティ（Gradient Penalties）（Kodali et al., 2017; Gulrajani et al., 2017a; Mescheder et al., 2018）
- 再構成損失（Reconstruction Loss）（Che et al., 2016）
- その他のヒューリスティック手法（Sønderby et al., 2016; Salimans et al., 2016; Arjovsky & Bottou, 2017; Berthelot et al., 2017）

さらに、タスクに特化したアーキテクチャ設計により、性能を大幅に向上させることも可能である（Radford et al., 2015; Karras et al., 2017）。

本研究では、識別器の正則化を通じて生成器に対するフィードバックを改善することを目指す。しかし、勾配の明示的な正則化やアーキテクチャ特有の制約を設けるのではなく、識別器に一般的な情報ボトルネック（Information Bottleneck）を適用する。

過去の研究では、情報ボトルネックがネットワークに不要な手がかり（Irrelevant Cues）を無視させる効果を持つ ことが示されている（Achille & Soatto, 2017）。本研究では、これにより生成器が本物と偽物の識別に最も重要な違いに焦点を当てられるようになる と仮定する。

敵対的手法（Adversarial Techniques）は、逆強化学習（Inverse Reinforcement Learning, IRL） にも応用されており（Fu et al., 2017）、デモンストレーションから報酬関数を回復し、その報酬関数を用いてポリシーを学習し、特定のスキルを再現できるようにする。

Finn et al. (2016a) は、最大エントロピー IRL（Maximum Entropy IRL）と GAN の等価性 を示した。また、敵対的模倣学習（Adversarial Imitation Learning） の手法も開発されており（Ho & Ermon, 2016; Merel et al., 2017）、報酬関数を明示的に回復することなく、エージェントがデモンストレーションを模倣する ことを可能にしている。

敵対的手法の利点 の一つは、識別器（Discriminator）を報酬関数の代替として活用することで、設計が困難な報酬関数を必要とせずにスキルの模倣が可能になる ことである。

しかし、敵対的手法を用いて学習されたポリシーの性能は、手作業で設計された報酬関数を用いた場合と比べると依然として劣る（Rajeswaran et al., 2017; Peng et al., 2018）。

本研究では、従来の敵対的手法を大幅に改善できることを示し、手作業で設計された最先端の報酬関数を用いた手法と同等の品質の結果を達成する ことを実証する。



本研究で提案する変分識別器ボトルネック（Variational Discriminator Bottleneck, VDB） は、情報ボトルネック（Information Bottleneck） に基づいている（Tishby & Zaslavsky, 2015）。これは、内部表現を正則化し、入力との相互情報量（Mutual Information）を最小化する手法 である。

直感的には、圧縮された表現は、元の入力に含まれる無関係なノイズを無視することで、汎化性能を向上させる 効果を持つ。

情報ボトルネックは、変分境界（Variational Bound）と再パラメータ化トリック（Reparameterization Trick） を活用することで、実際の深層モデルに適用可能 となる。このアプローチは、変分オートエンコーダ（VAE） の手法（Kingma & Welling, 2013）から着想を得たものであり、深層ネットワーク内での圧縮効果を近似 する（Alemi et al., 2016; Achille & Soatto, 2017）。

また、類似のボトルネック手法は、独立した特徴表現（Disentangled Representations） を学習するためにも応用されている（Higgins et al., 2017）。


VAE と GAN の組み合わせに関する研究 も進められており、多くの試みがなされている。



- Makhzani et al. (2016) は、VAE の学習時に敵対的識別器を導入 し、潜在エンコーディングの周辺分布を事前分布に近づける手法を提案した。
- Mescheder et al. (2017) や Chen et al. (2018) も、類似の手法を用いている。
- Larsen et al. (2016) は、GAN の生成器を VAE によってモデル化 するアプローチを採用した。
- Zhao et al. (2016) は、VAE の代わりにオートエンコーダを識別器のモデル化に使用 したが、エンコーディングに対する情報ボトルネックの制約は課していない。


近年の敵対的学習（Adversarial Learning）では、インスタンスノイズ（Instance Noise） が広く用いられている（Salimans et al., 2016; Sønderby et al., 2016; Arjovsky & Bottou, 2017）。

しかし、本研究では、単にノイズを加えるだけでなく、明示的に情報ボトルネックを適用することで、さまざまな応用領域において性能が向上することを示す。


## Preliminaries

$$
\begin{align}
J(q, E) = \min_{q,E} \mathbb{E}_{x,y \sim p(x,y)} \mathbb{E}_{z \sim E(z|x)} [-\log q(y|z)] \\
I(X, Z) = \int p(x, z) \log \frac{p(x, z)}{p(x)p(z)} \,dx\,dz
        = \int p(x)E(z|x) \log \frac{E(z|x)}{p(z)} \,dx\,dz.
\end{align}
$$



























