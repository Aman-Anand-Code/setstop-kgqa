# SetStop-KGQA

## Adaptive Graph Search for Multi-Answer Knowledge Graph Question Answering

SetStop learns which graph branch to explore next and when to stop
searching for additional answers. It combines a gain controller with
an answer-node recognizer and is evaluated on selected multi-answer
subsets of WebQuestionsSP (WebQSP) and ComplexWebQuestions (CWQ).

The project studies adaptive search under a traversal budget. It does
not claim state-of-the-art KGQA accuracy or guaranteed answer-set
completeness.

**Paper status:** Manuscript in preparation.

## Motivation

A question can have several correct answers. Finding one answer does
not mean that the full answer set has been recovered.

Continuing to explore the graph may discover missing answers, but it
can also introduce false positives and increase traversal cost.
SetStop learns to allocate search effort according to the current
state instead of assigning every question the same fixed action budget.

## Method

An action is a `(source entity, relation)` branch. Expanding it retrieves
the corresponding outgoing edges and target entities.

The search procedure is:

1. Initialize the search from the supplied topic entities.
2. Construct and shortlist the available frontier actions.
3. Predict a gain class for each action: `0`, `1`, `2`, or `3+`.
4. Rank actions using their predicted gain probabilities.
5. Stop if no shortlisted action is sufficiently likely to recover
   an additional answer, or if the frontier is empty or the action
   cap has been reached.
6. Otherwise, expand the best action, recognize answers among its
   target entities, update the search state, and repeat.

Offline gain labels count previously unseen gold answers reachable
through an action within the remaining hop allowance. They describe
reachable potential, not necessarily the answers obtained immediately
by one expansion.

Gold answers are used for supervision and evaluation. They are
unavailable to the learned search policy at inference time.

### Frontier ranking

The ranking score is:

`score = P(gain=1) + 2 × P(gain=2) + 3.5 × P(gain=3+)`

The value `3.5` is a fixed heuristic weight for the open-ended class.
It is not learned or calculated as an empirical mean. Consequently,
this score is not an exact expected answer count.

The same weights are used by the learned fixed-budget ranker and
SetStop on both datasets. No completed sensitivity analysis is
claimed for this choice.

### Model components

| Component | Model | Training |
|---|---|---|
| Gain controller | Qwen2.5-0.5B-Instruct | Four-way classification; four-bit backbone with rank-8 LoRA |
| Answer recognizer | MiniLM-L12-H384-uncased | Binary answer-node classification |

Each controller uses 60,000 balanced training candidates and one
training epoch. Answer recognizers are trained for two epochs.

Separate models are trained for WebQSP and CWQ. The CWQ experiment
is not zero-shot transfer from a WebQSP checkpoint.

## Repository Organization

| Folder | Contents |
|---|---|
| `diagram/` | Final method and result figures |
| `notebook/` | Numbered Colab notebooks |
| `data/` | Compact result reports, frozen settings, and calibration files |

The repository root contains `README.md`, `.gitignore`, and
`requirements.txt`.

Large datasets, generated training states, and model checkpoints are
stored separately from the repository.

## Datasets

We use the question-specific graphs distributed with the
[NSM authors' preprocessing package](https://github.com/RichardHGL/WSDM2021_NSM).

| Selected dataset | Training questions | Development questions | Test questions |
|---|---:|---:|---:|
| WebQSP | 1,200 | 118 | 736 |
| CWQ | 4,644 | 520 | 495 |

Questions must have valid topic entities, 2–50 mapped gold answers,
and every gold answer present in the supplied graph. CWQ additionally
requires answer reachability within four hops. One selected CWQ
training question matching held-out question text was excluded.

These results apply to selected multi-answer subsets, not the
complete official benchmarks. Entity linking and initial graph
construction are supplied by the preprocessing package.

### Dataset files

The notebooks use:

- `train_simple.json`
- `dev_simple.json`
- `test_simple.json`
- `entities.txt`
- `relations.txt`

The Colab project root is:

`/content/drive/MyDrive/SetStop_KGQA/`

Extracted datasets are stored under:

- `data/extracted/webqsp/`
- `data/extracted/cwq/`

These are Colab runtime paths, separate from the GitHub `data/`
folder containing experiment reports.

## Frozen Settings

| Setting | WebQSP | CWQ |
|---|---:|---:|
| Maximum graph hops | 3 | 4 |
| Frontier shortlist limit | 16 | 128 |
| Maximum actions | 12 | 24 |
| Controller threshold | 0.710 | 0.853 |
| Answer threshold | 0.670 | 0.960 |
| Matched fixed-ranker action cap | 6 | 14 |

Thresholds are selected using development data and remain frozen
during test evaluation.

The WebQSP cap-6 comparison is a retrospective mean-cost reference
for the frozen operating point. CWQ cap 14 is selected from
development mean actions. Matching is approximate in mean actions;
it does not enforce identical per-question or edge costs.

## Results

Scores are reported on a 0–1 scale. Actions and edges are means
per question.

### WebQSP: 736 Selected Test Questions

| Method | Macro F1 | Exact match | Complete recall | Actions | Returned edges |
|---|---:|---:|---:|---:|---:|
| Relation fixed, cap 12 | 0.3017 | 0.1128 | 0.4062 | 12.00 | 148.56 |
| Learned ranker, cap 6 | 0.3030 | 0.1318 | 0.3832 | 6.00 | 105.47 |
| Learned ranker, cap 12 | 0.3384 | 0.1277 | 0.4592 | 12.00 | 171.04 |
| SetStop, maximum cap 12 | 0.3856 | 0.2772 | 0.3927 | 5.95 | 110.28 |

### CWQ: 495 Selected Test Questions

| Method | Macro F1 | Exact match | Complete recall | Actions | Returned edges |
|---|---:|---:|---:|---:|---:|
| Relation fixed, cap 14 | 0.2968 | 0.1697 | 0.3758 | 14.00 | 127.58 |
| Relation fixed, cap 24 | 0.3512 | 0.1960 | 0.4505 | 24.00 | 188.93 |
| Learned ranker, cap 14 | 0.2998 | 0.1818 | 0.3818 | 14.00 | 94.13 |
| Learned ranker, cap 24 | 0.3511 | 0.2040 | 0.4646 | 24.00 | 150.17 |
| SetStop, maximum cap 24 | 0.3426 | 0.2545 | 0.3879 | 14.12 | 91.38 |

### Approximately Matched Mean Actions

SetStop is compared with the same learned ranker and answer
recognizer at approximately matched mean action counts.

| Dataset | Macro-F1 gain, percentage points | Paired 95% confidence interval |
|---|---:|---|
| WebQSP | +8.26 | [+6.27, +10.36] |
| CWQ | +4.28 | [+1.86, +6.86] |

Intervals use 5,000 paired question-level bootstrap samples.
They do not measure variability across training seeds.

Complete-answer recall gains at these matched budgets are not
statistically conclusive.

### Full-Graph Reference Models

Released NSM+h and ReaRev checkpoints were evaluated on the same
736 selected WebQSP questions.

| Method | Macro F1 | Exact match | Complete recall | Mean input edges |
|---|---:|---:|---:|---:|
| NSM+h | 0.7570 | 0.5082 | 0.6889 | 4,619.74 |
| ReaRev | 0.8123 | 0.5054 | 0.6522 | 4,619.74 |

These models remain substantially more accurate. SetStop's
contribution is adaptive search control under a traversal budget.

Search-policy counts measure returned edges, whereas full-graph
reference counts describe their input graphs. Neither measurement
includes graph loading, metadata access, or neural scoring.
The results therefore do not establish lower latency or FLOPs.

## Reproduction

Open the notebooks in Google Colab and execute them in their
numbered order.

| Stage | Runtime |
|---|---|
| Dataset download and audit | CPU |
| State generation and integrity checks | CPU |
| Gain-controller training and calibration | T4 GPU |
| Answer-recognizer training and calibration | T4 GPU |
| Development online evaluation | T4 GPU |
| Frozen test evaluation | T4 GPU |
| Cost-matched budget ablations | T4 GPU |
| Result aggregation and paper analysis | CPU |

Follow the installation cells in each notebook. The recorded runs
use PyTorch 2.11.0, Transformers 5.17.0, and a Tesla T4 GPU.

Keep the working Colab CUDA/PyTorch installation. The core CWQ
dependency versions are recorded in `requirements.txt`; this file
is not a complete environment lock.

Legacy NSM/ReaRev runners have separate compatibility requirements.

## Experiment Reports

The `data/` folder contains:

- `main_results.csv`: verified result rows.
- `frozen_settings.json`: model and search settings.
- `matched_budget_comparisons_v1.json`: paired comparisons and intervals.
- `two_dataset_paper_summary_v1.json`: consolidated experiment summary.

Development calibration reports can also be included:

- `dev_stop_threshold_v1.json`
- `answer_recognizer_threshold_v1.json`
- `cwq_dev_stop_threshold_v1.json`
- `cwq_answer_recognizer_threshold_v1.json`

Oracle results in the summary reports are diagnostic and are not
deployable learned-policy results.

## Limitations

- Evaluation uses selected subsets with gold-answer graph coverage.
- Frontier shortlisting and search limits can exclude useful branches.
- Adaptive stopping does not guarantee complete answer sets.
- Answer-recognition errors affect predictions and subsequent search.
- Offline gold-recognized counts differ from online learned counts.
- The open-ended gain weight is a heuristic.
- Experiments use one training seed and separate models per dataset.
- Returned-edge counts are a structural proxy, not total inference cost.

## Acknowledgments

This project uses the following public resources:

- [NSM implementation and preprocessed graphs](https://github.com/RichardHGL/WSDM2021_NSM)
- [ReaRev implementation](https://github.com/cmavro/ReaRev_KGQA)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
- [WebQSP dataset paper](https://aclanthology.org/P16-2033/)
- [ComplexWebQuestions dataset paper](https://aclanthology.org/N18-1059/)

Third-party datasets, code, and models retain their respective terms.

## Paper Citation

The manuscript is in preparation. The final author list, paper link,
and citation will be added when available.
