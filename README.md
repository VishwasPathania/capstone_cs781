CS781 Capstone — In-House Legal Intelligence System
Indian Institute of Technology Kanpur (IIT Kanpur)
Author: Vishwas Pathania (231168)
Course: CS781 Capstone Project
Repository: VishwasPathania/capstone_cs781
Overview
This repository contains the notebooks for a seven-phase, self-hosted legal NLP capstone focused on Indian legal documents. We pretrained a family of language models from scratch on the supplied legal corpus and adapted our own checkpoints to six downstream tasks. The work covers data preparation, model architecture, full/partial fine-tuning, parameter-efficient fine-tuning, knowledge distillation, evaluation, and competition submission generation.

The goal is to explore practical, locally deployable legal-language modeling under limited GPU resources and without external inference APIs. The phases are separate task-specific systems, not seven stages of a single end-to-end inference pipeline. Results are experimental and should not be used as legal advice.
Repository organization
capstone_cs781/
├── README.md
├── phase1/                      # Foundation-model pretraining
│   └── *.ipynb                  # GQA, SWA, MLA, encoder–decoder, hybrid LM, etc.
├── phase2/                      # Legal named entity recognition
│   └── *.ipynb
├── phase3/                      # Rhetorical role prediction
│   └── *.ipynb
├── phase4/                      # Court judgment prediction
│   └── *.ipynb
├── phase5/                      # Legal statute identification
│   └── *.ipynb
├── phase6/                      # Abstractive summarization
│   └── *.ipynb
└── phase7/                      # Prior case retrieval
    └── *.ipynb
Each phase folder contains its individual experimental submissions as separate notebooks. The descriptions below explain what each model family or submission variant implements. Notebook names may differ: use the actual files present in each folder to identify the variant. Some notebooks include multiple experimental stages or submission-generation cells.
Official results
Scores below are from Codabench; different phases have different metrics, so scores should not be compared across rows as if they represent the same task.
Phase	Task	Selected approach	Official score	Rank	Metric
1	Legal language modeling	Sliding-window attention (SWA)	0.8326	57	BERTScore F1
2	Legal named entity recognition	GQA + BiLSTM + CRF/span correction	0.5827	28	Strict span macro-F1
3	Rhetorical role prediction	MLM encoder + LoRA + document BiLSTM	0.5372	36	Macro-F1 across domains
4	Court judgment prediction	Fully fine-tuned encoder v2 ensemble	0.7620	36	Macro-F1
5	Legal statute identification	Fully fine-tuned GQA + calibration	0.3997	36	Macro-F1
6	Abstractive summarization	SWA + LoRA	0.8388	68	BERTScore F1
7	Prior case retrieval	MLA + QLoRA + lexical fusion/calibration	0.4400	41	Micro-F1@7


42 valid competition submissions were recorded in total. Ranks were recorded on 9 November 2026. The reported Phase-6 official score (0.8388) is distinct from the 0.8380 variant score in the report's comparison table; both values are preserved in their respective contexts.
Shared methodology
1. Data: Use the course-provided Indian legal datasets. Phase 1 draws on Acts, High Court judgments, Supreme Court judgments, and Tribunal decisions. Downstream phases use their corresponding task datasets and prescribed splits.
2. Foundation training: Train tokenizer and model weights from scratch, without importing pretrained external LM weights. The Phase-1 model family supplies checkpoints for downstream experiments.
3. Transfer: Initialize downstream backbone weights from our own checkpoints and add new task-specific heads. Compare full/partial fine-tuning, LoRA-family methods, adapters, and/or distillation where applicable.
4. Long-document processing: Bound attention cost by truncating, selecting head/tail evidence, chunking long texts, or aggregating multiple chunk representations.
5. Model selection: Run local validation and controlled or staged ablations. Where applicable, select checkpoints, decoding strategies, thresholds, ensembling, or retrieval calibration on development data.
6. Submission: Format final predictions exactly as required by the respective competition task, preserving document IDs and output schemas.
Phase-1 model family
The shared tokenizer is a 50,000-entry byte-level BPE trained on a large sampled legal corpus. The selected SWA model has 139,088,640 parameters, 16 Transformer blocks, model width 768, 12 query heads and 4 key/value heads, 64 dimensions per head, SwiGLU FFN width 2,048, pre-RMSNorm, RoPE, and tied token embeddings/output weights. It alternates 128-token causal local attention with global attention at layers 4, 8, 12, and 16. Its pretraining context is 1,024 tokens.

Other Phase-1 variants investigate different attention patterns and objectives. A separate bidirectional masked-language-model (MLM) encoder was pretrained later using the Phase-1 corpus and reused for several downstream tasks; it was not one of the five official Phase-1 submitted models.
Phase 1 — Legal language model pretraining
Task: Generate a 128-word continuation from a 512-word legal-text prompt, evaluated with BERTScore F1.

Data and preprocessing
- Source corpus: 3,480,245 legal documents from Acts, High Courts, Supreme Court, and Tribunals.
- Apply Unicode NFKC normalization, whitespace/signature-line cleaning, and SHA-256 deduplication within each source while preserving legal references and punctuation.
- Construct a 2% held-out document split per source (seed 42), train the byte-BPE tokenizer on training text, and pack EOS-separated token sequences into memory-mapped uint16 shards.
- Balance source sampling approximately in proportion to the square root of source token count rather than allowing the largest corpora to dominate.
Individual architecture/submission notebooks
Variant	Architecture and distinguishing implementation	Reported official score
GQA	Decoder-only causal Transformer with grouped-query attention; 12 query heads share 4 KV heads across 16 global-attention layers; 512-token training context.	0.8304
SWA	16-layer decoder with 128-token local attention plus full-attention layers 4/8/12/16; 1,024-token context. Selected Phase-1 model.	0.8326
MLA	Decoder-only Transformer with multi-head latent attention and compact latent KV representation of width 192; 512-token context.	0.8293
Encoder–decoder	Six encoder layers, eight decoder layers, cross-attention, and suffix prediction; 704-token source + 320-token decoder-side allocation.	0.8300
Hybrid prefix LM	Smaller width-512 model with document-isolated prefix attention, 12 local/four global layers and suffix-only loss; 704-token prefix.	0.8292


The selected SWA run processed about 500.171 million training tokens, used next-token cross-entropy, AdamW, FP16, gradient accumulation and cosine learning-rate decay. Its selected checkpoint at step 15,000 achieved validation loss 1.440508 / perplexity 4.2228. Submission generation uses the last 224 prompt tokens, temperature 0.7, nucleus sampling top_p=0.95, repetition penalty 1.10 and KV caching. Generation stops at 128 words or a 288-generated-token ceiling; therefore some outputs may be shorter than 128 words.

What to inspect in the notebooks: tokenizer construction; training/validation memmaps; attention implementation and Q/K/V tensor layouts; causal/local/prefix masking; training loops and saved checkpoints; local decoding diagnostics; final ID-aligned continuation file.
Phase 2 — Legal named entity recognition
Task: Identify 12 entity types as exact, end-exclusive character spans (start, end, label) in legal documents. Metric: strict span-level macro-F1.

Architecture: A pretrained causal backbone provides contextual token states. A two-layer bidirectional LSTM (384 hidden units per direction) supplies right-context information unavailable to a pure causal decoder. Concatenated backbone/LSTM features feed a 25-class BIO tagger and two 41-way byte-offset prediction heads (values 0–40 for token-boundary corrections). A CRF learns valid tag transitions; Viterbi decoding, UTF-8 byte-to-character alignment, overlapping-window merging and selected boundary/person rules recover exact original-text spans.

Implementation: Preserve original text and character offsets; tokenize into 512-token windows with stride 384 and central 64-token margins; train with weighted BIO cross-entropy, normalized CRF negative log-likelihood, and half-weighted start/end offset losses. The local dev split uses 56 of 70 labeled documents, followed by refitting and multiple seeds.

Individual submissions: SWA without BiLSTM/rules (0.3892), SWA + BiLSTM before rules (0.4773), SWA + BiLSTM + rules (0.5749), partial SWA fine-tuning (0.5254), GQA (0.5827), MLA (0.5618), and an encoder–decoder-derived distilled student (0.4065). The final GQA system includes the sequence/span reconstruction pipeline, not just a BIO classifier.
Phase 3 — Rhetorical role prediction
Task: Assign one of 13 rhetorical-role classes to each sentence, retaining the original sentence order and document alignment across Corporate Law and Income Tax domains. Metric: domain-averaged macro-F1.

Architecture: Encode a target sentence with neighboring context (up to 256 tokens); combine mean and learned attention pooling into a 512-dimensional sentence representation. Add a 16-dimensional domain embedding, numeric relative-position features, and learned position buckets. A two-layer document BiLSTM operates over 32-sentence windows with stride 16, then predicts per-sentence roles. Aggregate overlapping-window predictions and use a prior-adjusted decode; the selected system ensembles three seeds.

Approach: Document structure matters: neighboring sentences often share rhetorical roles, and decision/ruling sentences frequently occur near the end. The final MLM encoder + LoRA system tunes a small subset of the encoder and the document-level classifier with weighted, smoothed cross-entropy.

Individual submissions: GQA + LoRA v1/token BiLSTM (0.4099); GQA + LoRA v2/hierarchical context (0.4852); encoder + LoRA (0.5372); encoder + LoRMA (0.4987); encoder–decoder-derived KD (0.4483); SWA + AdaLoRA (0.5303); encoder + AdaLoRA (0.5225).
Phase 4 — Court judgment prediction
Task: Predict one binary judgment outcome per case. Metric: macro-F1.

Architecture: Encode selected case segments into chunk representations, combine token-level mean pooling with distance-aware document attention, and preserve a final-chunk shortcut to capture outcome evidence. The final encoder v2 uses eight 512-token chunks per case and an ensemble. Scores are converted to labels using a top-half-positive rule motivated by the development-set class prior.

Methodology: Sequential probes compared head, tail, combined head/tail and compact-tail input; mean versus learned token attention; and single versus multi-chunk coverage. The improved v2 jointly changes lowercasing, coverage, regularization, exponential moving average (EMA) and rank ensemble. It is a bundled revision, not a single-factor ablation.

Individual submissions: Encoder full fine-tuning v1 (0.7440); encoder full fine-tuning v2 (0.7620); GQA full fine-tuning without BiLSTM (0.7198); encoder partial fine-tuning of top eight layers (0.7488); encoder–decoder-derived KD student (0.7384); SWA + LoRMA (0.6275).

Data: CJPE; 37,387 train, 1,988 dev and 1,517 test rows before task-specific filtering. Cases are long, so input coverage and selection materially affect results.
Phase 5 — Legal statute identification
Task: Predict all applicable IDs from 100 statutes for each case. Metric: macro-F1 (multi-label classification).

Architecture: Two selected 512-token views per document enter a pretrained GQA backbone. The chosen head + cue-rest strategy retains introductory facts and evidence-rich later segments mentioning Sections, Acts or Articles. Token pooling combines closing-EOS and attention representations; a 100-logit classification head predicts multiple labels. A learned label-interaction rescorer and per-label threshold shrinkage calibrate the final output.

Training: Weighted binary cross-entropy accounts for class imbalance, with staged searches over learning rate, loss, pooling and document views. The final GQA run restores epoch 4. Development calibration includes cross-fitting; exact-set match remains difficult even when per-label F1 improves.

Individual submissions: Encoder full fine-tuning (0.3821); encoder partial fine-tuning/top six (0.3769); GQA full fine-tuning (0.3892 in the variant table; official phase result 0.3997); SWA bottleneck adapters (0.3846); encoder–decoder-derived KD (0.3701); GQA + LoRMA (0.3771).

Data: 42,750 training cases, 10,181 dev cases and 13,019 test cases; approximately 3.7 applicable statute IDs per case on average.
Phase 6 — Abstractive legal summarization
Task: Generate a concise abstractive summary aligned to each source judgment ID. Metric: BERTScore F1.

Architecture: The selected solution adapts the SWA causal decoder using rank-16 LoRA on attention projections plus trainable normalization layers. Each 2,048-token training example allocates about 1,530 source tokens, separator/BOS markers, and 512 summary-target positions. Loss is applied to summary tokens only. Source extraction chooses informative spans from long judgments before generation.

Methodology: Compare an encoder–decoder with a learned chunk compressor against SWA, GQA and MLA alternatives. SWA experiments examine extracted versus head-only sources, context length/RoPE handling, target budget, adaptation type and decoding. Final generation uses greedy decoding with repetition penalty 1.2, no-repeat 4-gram blocking, 16–512 new tokens, sentence deduplication and tail trimming.

Individual submissions: Encoder–decoder compressor-only (0.7995); SWA + Houlsby adapters (0.8372); GQA + LoRMA (0.8072); MLA sequence-level KD student (0.78444); SWA + LoRA (0.8380 in the variant table; official phase score 0.8388).

Data: 7,030 supplied train document–summary pairs, reduced to 7,001 cleaned distinct pairs before holdout/refitting; 100 private-reference test documents. Long-input coverage is a fundamental bottleneck; naive full-document map–reduce did not outperform the selected single-pass strategy.
Phase 7 — Prior Case Retrieval (PCR)
Task: Rank and return seven distinct candidate-case IDs for every query judgment. Metric: micro-F1@7.

Architecture: Encode query and candidate documents using a shared-weight bi-encoder. The selected MLA + QLoRA model uses a 16-layer, width-768 latent-attention backbone with latent KV width 192, frozen 4-bit NF4 linear weights, and rank-16 LoRA adapters (alpha=32). Process long inputs as overlapping 512-token chunks and spread an inference selection of at most 48 chunks across each document. L2-normalized chunk vectors support symmetric best-chunk/MaxSim similarity.

Training and ranking: Sample positive cases and BM25 hard negatives; use InfoNCE contrastive learning on mean-vector and late-interaction similarities (temperature 0.05), masking other known positives. Fuse neural scores with lexical unigram/bigram BM25, statute references and document-head evidence. Normalize scores per query, apply candidate-wise calibration, and return the top seven unique IDs. Candidate-pool calibration uses unlabeled test pool information, not private relevance labels.

Individual submissions: GQA full fine-tuning v1 (0.3272); encoder full fine-tuning v1 (0.2930); GQA full fine-tuning v2 (0.4117); SWA + LoRMA (0.4146); MLA + QLoRA (0.4400); encoder–decoder-derived KD (0.4123); GQA partial fine-tuning/top six layers (0.4371).

Data: Train 827 queries / 4,320 candidates; dev 118 / 1,023; test 237 / 1,727. Training excludes ten queries lacking positives. Neural-only retrieval was weaker than the final combined neural/lexical/calibrated pipeline; the final local MLA F1 moved from 0.3535 neural-only to 0.4503 fully calibrated (and 0.4400 on the official test).
Experimental setup and reproducibility
Environment
The notebooks were developed for GPU-backed Google Colab/Kaggle-style environments (some logs explicitly identify a Kaggle NVIDIA T4). Hardware, batch size, checkpointing and memory requirements vary by phase and notebook. Models use PyTorch with mixed precision, gradient accumulation, and, where useful, activation checkpointing. A CUDA GPU and substantial storage are strongly recommended, especially for Phase-1 pretraining and long-document phases.

Common dependencies include Python 3, PyTorch, NumPy, pandas, scikit-learn, Hugging Face tokenizers/Transformers where used, and phase-specific packages for metrics, PEFT/LoRA, quantization, or retrieval. Install the exact packages/versions specified by each notebook's setup cells. This repository description does not claim a single pinned environment or universal requirements.txt, because the seven phases use different stacks.
How to run a notebook
1. Clone the repository and open the specific .ipynb from the relevant phase folder in Colab, Kaggle, or Jupyter with an appropriate CUDA environment:
   git clone https://github.com/VishwasPathania/capstone_cs781.git
   cd capstone_cs781
   # Example: launch your Jupyter installation, then open phase1/<notebook>.ipynb
   jupyter lab
2. Download the course-provided datasets corresponding to that phase. The report identifies the dataset source as Exploration-Lab/CS-781-Capstone; access and distribution may be subject to the course's permissions. Preserve the expected Train/Dev/Test file structure.
3. In the notebook's configuration/setup cells, update dataset roots, output directories, GPU settings, pretrained checkpoint paths and cache paths to match your environment. Do not assume Google Drive/Kaggle mount paths transfer unchanged.
4. Run preprocessing/tokenization and cache-generation cells before model training, unless the notebook explicitly loads an existing compatible cache.
5. For Phase 1, train the tokenizer and selected architecture from random initialization. For Phases 2–7, load the requisite locally pretrained Phase-1 or later MLM-encoder checkpoint, and execute the phase-specific training/adaptation cells. KD notebooks additionally require their own trained teacher checkpoints or cached teacher outputs.
6. Run local validation and checkpoint selection, then refit if the notebook's documented submission path requires it. Finally run the test inference/submission-generation cells and validate the output schema before uploading to Codabench.
Important: Notebooks are research submissions, not one-click scripts. Large checkpoint files, dataset paths, competition credentials and trained weights are not guaranteed to be included in the GitHub repository. Reproducing the numerical scores requires the correct data, checkpoint lineage, seeds, environment and notebook-specific execution order. Consult each notebook's parameter cells rather than assuming every submission is runnable independently from a fresh checkout.

Phase-to-phase checkpoint dependencies
Supplied legal corpus
  ├─> Phase 1: tokenizer + GQA / SWA / MLA / encoder–decoder / hybrid checkpoints
  │     └─> Phase 2–7: separate task-specific fine-tuning / adapters / KD experiments
  └─> Later masked-LM encoder pretraining
        └─> Encoder-based Phase 3–5 and Phase 7 experiments
Phase numbers indicate competition tasks; they do not imply that inference on Phase 7 must run Phases 2–6 first.
Representative selected settings
Phase	Context or input selection	Optimization / inference summary
1	1,024 tokens	AdamW, peak LR 2e-4, batch 4 × accumulation 8; 15,259 planned steps; nucleus sampling
2	512-token windows	BiLSTM + CRF + offset heads; three-seed refitting; span correction
3	256-token sentence context; 32-sentence windows	LoRA/head training; three-seed ensemble; prior-adjusted roles
4	8 × 512-token chunks	Fully tuned encoder v2; document attention; top-50% class assignment
5	2 × 512-token selected views	Weighted BCE; best epoch 4; per-label threshold calibration
6	2,048-token source+target sequence	SWA LoRA; summary-only cross-entropy; greedy controlled decoding
7	Up to 48 × 512-token inference chunks	MLA QLoRA; contrastive learning; neural + lexical calibrated top-7


The selected configurations are not standardized compute budgets. Results from staged probes, final refits and official submissions may involve different training durations and evaluation datasets.
Limitations and interpretation
- The initial pretraining budget was capped around 500M tokens for the selected model, rather than exhausting the complete legal corpus.
- Some ablations are short, single-seed or sequential configuration searches; the score change is not necessarily attributable to one architectural component.
- Long-document truncation and chunk caps can omit decisive evidence. Rare entity, role and statute classes remain challenging.
- BERTScore/ROUGE do not establish summary factuality; retrieval and prediction F1 do not establish legally valid reasoning or citations.
- Dev-selection and refitting choices can create local-to-hidden-test differences. An independent legal-expert evaluation and source-grounded verification would be required before consequential deployment.
