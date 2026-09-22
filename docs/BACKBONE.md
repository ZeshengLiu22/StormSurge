# Backbone and representation

`PACT` and `Baseline` are assembled by [`build_model`](../emulator/models/architectures.py). The CLI name `--model perceiver3` constructs `ModelConfig(model="pact")`; `--model baseline` constructs the baseline and selects its single linear head. This document follows the representation up to the output-head interface. The task and timestamp contract are defined in [FORMULATION](FORMULATION.md).

<!-- choices encoder_type: GraphSAGE,CNN -->
<!-- choices temporal_block: MLP,LSTM,GRU,Transformer -->

## Dimensions and configuration

Let $B$ be the batch size, $N_i$ the forcing nodes in graph $i$, $F$ the input feature count, $d$ the hidden width, $H$ the requested history in hours, $S=H/6+1$ the number of forcing slices, and $K$ the number of target horizons. The current aligned station data have $F=5$, $K=6$, and hourly targets. Grid dimensions are denoted $R,C$, independently of history $H$.

The parser and generated experiments have distinct defaults. [`arguments.py`](../emulator/training/arguments.py) defines direct CLI defaults; [`generate_configs.py`](../tools/generate_configs.py) supplies the canonical experiment values:

| Setting | Direct CLI default | Canonical generated experiments |
|---|---|---|
| Model | `baseline` | `perceiver3` / PACT |
| `hidden_channels` | 64 | 128 |
| `encoder_type`, `num_layers` | GraphSAGE, 2 | GraphSAGE, 2 |
| `history_hours` | 24 | 24, hence $S=5$ |
| `temporal_block` | Transformer | Transformer |
| `node_read_heads`, `time_read_heads` | 8, 8 | 8, 8 |
| `transformer_layers`, `transformer_ff_mult` | 2, 4 | 2, 4 |
| `dropout` | 0.05 | 0.05 |
| `transformer_dropout` | 0.05 | 0 |
| `head_dropout` | 0 | 0.05 |
| `max_time_steps` | 32 | 32 |
| `x_norm`, `x_clip`, `x_aug` | robust, 5, enabled | zscore, 0, disabled |
| Station elevation / bathymetry features | included / excluded | excluded / excluded |

The Python `ModelConfig` has its own construction defaults: explicit `hidden_channels`, PACT, two history steps, spatial dropout 0, and head dropout 0. Training fills this dataclass from resolved CLI settings. Hidden width must be divisible by each configured attention head count. `max_time_steps` must cover $S$, and the stored input must contain at least $S$ slices. `history_hours` must be a nonnegative multiple of six.

## Inputs, station features and normalization

[`ForcingGraphView`](../emulator/data/graph_store.py) slices stored `[available_steps,N,F]` history to the last $S$ steps and transposes it to `[N,S,F]`. Batching produces:

| Tensor | Shape | Meaning |
|---|---|---|
| `batch.x` | $[\sum_i N_i,F]$ | Current forcing slice |
| `batch.x_hist` | $[\sum_i N_i,S,F]$ | Oldest retained forcing through current forcing |
| `batch.edge_index` | $[2,E]$ | Directed edges, offset by graph during batching |
| `batch.batch` / `batch.ptr` | node assignments / graph boundaries | Separate the $B$ graphs |
| `batch.y` | $[B,K]$ | Physical target surge, used by training/evaluation |
| `batch.target_timestamps` | $[B,K]$ | Integer UTC seconds, used for target validation/evaluation |

The forward model reads forcing and graph tensors. Target values and timestamps do not enter its representation. The history samples are forcing fields at $t-H,t-H+6,\ldots,t$, not previous surge labels.

The supplied [NCEP forcing preprocessing](../preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py) concatenates `[WVX, WVY, PR, longitude, latitude]`, then replaces `PR` by its anomaly from the spatial mean over the local grid at each forcing time. The [CMIP6 forcing implementation](../preprocessing/forcing_cmip6.py) uses the same five-channel arrangement and pressure-mean removal. Longitude/latitude are node features in degrees before feature normalization. Wind and pressure channels retain their source input units; the model does not perform another physical-unit conversion. [`time_align_unified.py`](../preprocessing/time_align_unified.py) flattens the spatial grid in row-major order. GraphSAGE consumes the saved connectivity; preprocessing supports directed `grid4`, `grid8`, and fully connected graphs. CNN derives adjacency from grid convolutions.

Optional station metadata, shared across the station-specific batch, is encoded by [`station_features_from_json`](../emulator/data/station_metadata.py) in this order:

\[
[\mathrm{lat}/90,\ \mathrm{lon}/180,\ (\mathrm{elevation}/10),\
 \sin\phi,\cos\phi,\sin\lambda,\cos\lambda,\ (\mathrm{bathymetry}/10)].
\]

Parentheses mark independently optional entries; angles $\phi,\lambda$ are radians. The vector has $6+I_{\rm elevation}+I_{\rm bathymetry}$ entries: seven with CLI defaults and six in generated configs. Enabled fields must exist and be finite. `use_station_meta=0` omits this vector and its projection. Station metadata is passed only to PACT when a station is selected.

[`fit_statistics`](../emulator/data/stats.py) fits per-feature input statistics from current `graph.x` nodes in TRAIN graphs and applies the same statistics to all retained history slices. `zscore` uses the feature mean and $\sqrt{\mathrm{variance}+10^{-6}}$. `robust` uses center $(Q_{p_{lo}}+Q_{p_{hi}})/2$ and scale $(Q_{p_{hi}}-Q_{p_{lo}})/2$. `mag` uses center zero and scale $Q_{p_{hi}}(|x|)$. Percentile scales have a $10^{-6}$ floor; robust/magnitude fitting samples up to `x_nodes_per_graph` nodes per TRAIN graph, with nonpositive settings selecting 256. These feature percentiles only normalize inputs.

[`normalize_inputs`](../emulator/data/normalization.py) assigns fresh FP32 normalized `x` and `x_hist` tensors, optionally clips them, and optionally applies training-only per-feature scale/bias jitter shared across all nodes and history steps of a batch. A second clip follows jitter. Stored graphs remain unchanged. Separate TRAIN target means $\mu_h$ and positive scales $s_h$ normalize each output horizon; physical reconstruction is $\hat y_{ih}=\mu_h+s_h\hat z_{ih}$.

## Spatial encoders

[`SpatialEncoder`](../emulator/models/spatial.py) is shared across forcing times; one set of spatial weights processes each of the $S$ slices. Both choices map $[\sum_iN_i,F]$ to $[\sum_iN_i,d]$.

### GraphSAGE

There are exactly `num_layers` `torch_geometric.nn.SAGEConv` layers, at least two, with widths $F\to d\to\cdots\to d$. The constructor uses mean aggregation, a learned root projection, bias, no pre-projection, and no output L2 normalization. At layer $\ell$, the preactivation for node $v$ is

\[
 a_v^{(\ell)}=W_{\rm neigh}^{(\ell)}
       \operatorname{mean}_{u\in\mathcal N(v)}x_u^{(\ell-1)}
       +b^{(\ell)}+W_{\rm root}^{(\ell)}x_v^{(\ell-1)}.
\]

The neighbor set follows the supplied directed `edge_index`. Each layer, including the last, applies `LeakyReLU(0.1)` then dropout with probability `dropout`. There is no extra spatial normalization or residual block; the root projection is part of each SAGE convolution.

### CNN

There are exactly `num_layers` `nn.Conv2d` layers, at least one. Every layer uses a $3\times3$ kernel, stride 1, padding 1 and bias. Widths are $F\to c\to\cdots\to c\to d$, where `cnn_intermediate_channel` sets $c$, default 29. A one-layer CNN maps $F\to d$ directly. Every convolution, including the last, applies in-place `LeakyReLU(0.1)` then spatial dropout.

`grid_shape` requires positive uniform `grid_H/grid_W` metadata and exactly $RC$ nodes per graph. Inputs reshape to `[B,F,R,C]`; outputs reshape back to `[B*R*C,d]` in the original row-major node order. Padding preserves resolution. The CNN has no pooling, residual connection, or normalization inside its spatial stack, and does not use `edge_index`.

## Station readout

PACT constructs a learned station query $q\in\mathbb R^{1\times d}$, initialized to zero. If station metadata $m$ is enabled, it adds

\[
 q=q_{\rm learned}+\operatorname{LN}_{\rm affine}
       \left(W_2\operatorname{LeakyReLU}_{0.1}(W_1m+b_1)+b_2\right),
\]

where the metadata MLP widths are `station_feat_dim` $\to\max(16,d)\to d$. This query is expanded to `[B,1,d]`.

At each forcing time, `to_dense_batch` converts spatial outputs into `[B,N_max,d]` and a valid-node mask. `node_readout` is `nn.MultiheadAttention` with `node_read_heads` heads, learned Q/K/V and output projections, bias and zero attention dropout. Per head it computes

\[
 \operatorname{softmax}\!\left(\frac{QK^\mathsf T}{\sqrt{d/a}}+M\right)V,
\]

where $a$ is the head count and the key-padding mask $M$ excludes padded nodes. Queries come from $q$, and keys/values come from spatial node representations. Concatenated heads and the output projection yield `[B,1,d]`.

`F.layer_norm(station.float(), (d,))` explicitly converts this output to FP32 and normalizes its final dimension, with epsilon $10^{-5}$, no learned affine parameters and no query residual. Squeezing the query axis and stacking time slices yields `[B,S,d]` station history.

## Temporal representation

A learned `lag_embed` table has `[max_time_steps,d]` entries. For oldest-to-current index $j$, PACT adds embedding $S-1-j$. Thus lag zero always denotes current forcing and one lag represents six hours. Lag and horizon embeddings use the standard `nn.Embedding` normal initialization. The learned station token is initialized to zero; other ordinary layer parameters retain their PyTorch/PyG initializers.

All temporal options preserve `[B,S,d]`. `transformer_layers` supplies their depth, `transformer_ff_mult` supplies feed-forward width $f=\lfloor d\,\mathrm{mult}\rfloor$ where applicable, and `transformer_dropout` supplies temporal dropout. The parser accepts `attn` as an alias for `Transformer`; it does not construct an additional temporal architecture. History is ordered oldest to current, and no causal attention mask is added because all tokens are observed input history.

### Transformer

Each [`TemporalTransformer`](../emulator/models/temporal.py) layer contains an affine pre-normalization, self-attention with `time_read_heads`, a scaled residual, and a `TemporalMLP` sublayer:

\[
 z=\operatorname{LN}_{\rm affine}(x),\qquad
 u=x+e^{\gamma_a}\operatorname{Dropout}(\operatorname{MHA}(z,z,z)),
\]
\[
 x'=u+e^{\gamma_f}\operatorname{FF}(\operatorname{LN}_{\rm affine}(u)).
\]

Both $\gamma_a$ and $\gamma_f$ are independent learned scalar parameters initialized to $\log(0.1)$, so their positive residual gains initially equal 0.1. The feed-forward sequence is `Linear(d,f) → LeakyReLU(0.1) → Dropout → Linear(f,d) → Dropout`. Attention has temporal attention-weight dropout plus the explicit output dropout. The stack contains `transformer_layers` such blocks.

### MLP

Each `TemporalMLP` applies only the second residual equation above, with its own affine LayerNorm, feed-forward sequence and learned $\log(0.1)$ gain. It is pointwise across the history axis: this block does not mix different time slices. The subsequent horizon readout still attends across all $S$ output tokens. The stack depth and feed-forward width use the same settings as the Transformer.

### LSTM and GRU

`TemporalRNN` constructs `nn.LSTM(d,d,depth)` or `nn.GRU(d,d,depth)`, unidirectional and `batch_first=True`. The output sequence $r$, rather than only the final state, forms a residual:

\[
 x'=\operatorname{LN}_{\rm affine}(x+\operatorname{Dropout}(r)).
\]

PyTorch recurrent inter-layer dropout is enabled only for depth greater than one; the output dropout remains present for every depth. These blocks use post-residual affine LayerNorm and no learned residual gain. The feed-forward multiplier is unused.

## Horizon readout and stability

`horizon_embed` contains $K$ learned $d$-dimensional horizon queries, expanded to `[B,K,d]`. `forecast_readout` applies `nn.MultiheadAttention(d,time_read_heads)` with horizon queries and temporal memory as keys/values. Its attention dropout is zero, and all history tokens are valid; there is no padding mask at this stage. Each horizon attends to the full memory, producing `[B,K,d]` contexts.

For $H=0$, hence $S=1$, PACT explicitly adds the horizon embeddings back to the attention output so the single memory token does not erase horizon identity. It then always applies `F.layer_norm(context.float(), (d,))`, the same FP32, non-affine functional normalization used after node readout. This context tensor is the common interface to every PACT output head.

The two readout normalization sites explicitly force FP32. Temporal, station-metadata, and optional head-pooling normalization modules are standard affine `nn.LayerNorm`, epsilon $10^{-5}$; the repository does not implement a custom FP32 LayerNorm wrapper. Affine LayerNorm begins with unit weight and zero bias. Model execution uses the trainer's autocast setting where enabled; production head activation outputs explicitly convert to FP32. Dropout is disabled in evaluation. Single, direct Dual, and Severity–Shape use the same upstream PACT representation for the same backbone configuration.

## Baseline representation

`Baseline` uses the same `SpatialEncoder`, then `global_mean_pool` over each graph's nodes. For $H=0$, it encodes only `batch.x` and obtains `[B,d]`. For $H>0$, it encodes each retained forcing slice, stacks graph means into `[B,S,d]`, passes this sequence through a single-layer, unidirectional LSTM, and takes its last output. This LSTM is used regardless of the PACT `temporal_block` setting.

The baseline then applies `nn.Dropout(dropout)` and a single `nn.Linear(context_width,K)` output layer. The Python `ModelConfig.temporal_hidden` can override its positive-history LSTM width; otherwise that width equals $d$. The $H=0$ control retains width $d$. This field is a Python/checkpoint construction option, not a training CLI argument. Baseline uses neither station-query attention, station metadata, lag embeddings nor horizon-query attention.

## Head interface and parameter accounting

PACT passes `[B,K,d]` contexts to `SingleHead`, `ExceedanceHead`, `ExceedanceHead_Experiment`, or `SeverityShapeHead`, all defined in [`heads.py`](../emulator/models/heads.py). `SingleHead` applies `Linear(d,w) → LeakyReLU(0.1) → Dropout(head_dropout) → Linear(w,1)` separately to each horizon and returns normalized `[B,K]` predictions. Here $w=2d$, unless the Python/checkpoint `ModelConfig.head_hidden` override is specified; this override is not exposed by the training parser. The Dual formulations are described in [DUAL_EXCEEDANCE](DUAL_EXCEEDANCE.md) and [SEVERITY_SHAPE](SEVERITY_SHAPE.md). The engine reconstructs physical predictions after the head returns.

[`count_model_parameters`](../emulator/models/reporting.py) counts unique registered `Parameter` objects by `numel`, unwraps DataParallel/DDP, and reports total, trainable, non-trainable, head, backbone, and immediate head-child counts. `requires_grad` determines trainability. Buffers, including fixed thresholds and fixed gate probabilities, are excluded. Backbone count is total minus head count. Counting requires no forward pass.

`train.py` logs `[Parameters]`, and saves the full `model_parameters` dictionary in checkpoints and the run summary. `infer.py` logs the same counts and writes them into result metadata. For canonical $F=5,K=6,d=128,S=5$, six station features, two GraphSAGE layers and two Transformer layers:

| PACT head | Total/trainable parameters | Head parameters | Backbone parameters |
|---|---:|---:|---:|
| Single | 618,885 | 33,281 | 585,604 |
| Production direct Dual | 685,447 | 99,843 | 585,604 |

These counts follow the current modules; changing widths, station features, encoder, temporal block, or supported head presets changes the corresponding counts.
