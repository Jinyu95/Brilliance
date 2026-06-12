# Knowledge Graph Methodology

## Abstract

This document describes the current knowledge-processing method used in the
BRILLIANCE project (*Beam Research with Intelligent LLM-driven Accelerator
kNowledge and Computational Engine*). The system is designed to transform heterogeneous
accelerator-design materials into a subject-centered knowledge graph and a
multi-level semantic cluster map that can support retrieval, prompting, and
human inspection. The implementation is intentionally lightweight and local:
it does not depend on external vector databases or opaque embedding services.
Instead, it combines ontology-guided extraction, sparse TF-IDF subject
profiling, and cosine-similarity clustering to produce a release-safe
knowledge representation.

The current public repository ships only the distilled graph artifacts, not the
original external source materials. As a result, the public bundle preserves
learned knowledge structure while removing source-level provenance and
third-party content.

## 1. System Objective

The knowledge subsystem has three goals:

1. Convert accelerator-design materials into structured machine-readable facts.
2. Organize those facts into learned semantic clusters rather than folder-based
   or manually imposed branches.
3. Expose a public, release-safe representation that remains useful to the
   agent after the original corpus has been removed.

This design is especially important for accelerator applications, where useful
knowledge is distributed across design notes, reference tables, API examples,
and procedural rules rather than a single homogeneous document class.

## 2. Corpus Processing Pipeline

The processing pipeline implemented in
[agents/knowledge_corpus.py](/c:/Users/86186/Desktop/accdesign/agents/knowledge_corpus.py)
and
[agents/knowledge_extraction.py](/c:/Users/86186/Desktop/accdesign/agents/knowledge_extraction.py)
consists of six stages.

### 2.1 Source discovery and normalization

In private build mode, the corpus builder can discover Markdown files, Jupyter
notebooks, and PDF documents according to
[knowledge/manifest.json](/c:/Users/86186/Desktop/accdesign/knowledge/manifest.json).

Each source type is normalized differently:

- Markdown is segmented by heading structure and then chunked by character
  length.
- Notebooks are converted into textual chunks from markdown/raw cells, with
  optional code inclusion disabled by default.
- PDFs are segmented by table of contents when available, otherwise by page
  ranges.

Chunk filtering is then applied using source-profile rules from the manifest.
These rules can favor design-relevant sections and suppress less useful
sections before extraction.

### 2.2 Ontology-guided semantic extraction

The extraction layer is not a generic LLM summarizer. It is a deterministic,
ontology-guided parser with typed outputs. The ontology and taxonomy are stored
in:

- [knowledge/taxonomy.json](/c:/Users/86186/Desktop/accdesign/knowledge/taxonomy.json)
- [knowledge/ontology.json](/c:/Users/86186/Desktop/accdesign/knowledge/ontology.json)

The extractor produces the following fact classes:

- `measurement`
  via regular-expression patterns over known physical quantities
- `relation`
  via sentence-level verb pattern matching constrained by category rules
- `table_measurement`
  via structured markdown-table and linear-table parsing
- `table_relation`
  via attribute-to-concept relations extracted from tables
- `design_rule`
  via heuristic detection of prescriptive design statements
- `api_signature`
  via parsing of `jt.<function>(...)` usage signatures

The current implementation uses the following confidence policy:

- regex measurements: `0.7` if attached to a subject, otherwise `0.4`
- sentence relations: `0.75`
- markdown/linear table facts: typically `0.85-0.95`
- design rules: `0.65-0.8`
- API signatures: `0.9`

Only facts above the configured minimum confidence are retained. The default
threshold is `0.55`.

### 2.3 Subject assignment and indexing

Facts are reorganized around subjects rather than around documents. A subject is
identified by a `(subject_category, subject_label)` pair. Examples include:

- physics quantities such as emittance or circumference
- reference machines such as ESRF-EBS or APS-U
- lattice types such as 7BA, 9BA, or HMBA
- software APIs such as `KQUAD`, `SBEND`, or `gettune`

Measurements are linked to nearby concepts by local proximity. In the current
implementation, the nearest eligible concept within a 140-character window is
used when available. Relations are additionally re-indexed by their target
concepts when the target is a reusable physical concept rather than a software
API or reference machine.

The result is a subject-centered intermediate representation stored in
`subject_index`, from which clustering and prompt-context generation are built.

### 2.4 Graph construction

Before release-safe pruning, the builder creates a typed graph with nodes such
as:

- `source`
- `section`
- `chunk`
- `concept`
- `quantity`
- `parameter`
- `fact`

and edges such as:

- `supports`
- `states_fact`
- `about`
- `object`
- `measures`
- `accepts_argument`

This graph structure is useful during private builds because it preserves the
connection between textual evidence, extracted facts, and reusable concepts.

## 3. Subject Representation for Clustering

Clustering is performed at the subject level rather than directly at the chunk
or fact level. Each subject is represented by a sparse textual profile composed
from:

- the subject label (doubled for emphasis)
- the subject category (doubled for emphasis)
- labels and truncated summaries from the top 5 linked facts (80 chars max each)

Evidence strings and chunk text are deliberately excluded from profiles to
prevent generic measurement language from dominating the sparse vectors and
causing semantic mixing across unrelated domains.

### 3.1 Sparse TF-IDF feature construction

Subject profiles are tokenized with a lightweight alphanumeric tokenizer and
converted into sparse TF-IDF vectors. Tokens shorter than three characters and a
project-specific stopword list are removed.

The vector weight for token `t` in subject `i` is:

`tfidf(i, t) = tf(i, t) * idf(t) * w_i`

where `w_i` is a subject-level importance factor. This factor is not constant:
it increases for subjects that are both information-rich and physically
important.

### 3.2 Subject weighting

The current implementation uses:

`w_i = (1 + log(1 + fact_count_i)) * category_weight_i * source_mix_weight_i`

Two design choices are important here.

First, category priors intentionally favor accelerator-physics concepts over
software-only concepts. For example, the current weights are higher for
`physics_quantity` (`2.4`), `design_method` (`2.0`), and `lattice_type`
(`1.9`) than for `software_api` (`0.45`). This was introduced to reduce
software-reference dominance in the learned cluster map.

Second, source-kind priors favor design-oriented and physics-oriented materials
over software reference material. In the current implementation,
`design_reference`, `handbook`, and `lecture_notes` receive stronger priors than
`software_reference`.

## 4. Clustering Method

The cluster method stored in the graph metadata is
`category_partitioned_tfidf_kmeans`. This is a two-stage approach that first
partitions subjects by ontology category, then sub-clusters within each
partition using sparse TF-IDF cosine-similarity K-means.

### 4.1 Category partitions

Before any vector similarity is computed, subjects are grouped into five
semantic partitions based on their ontology category:

| Partition | Categories |
|---|---|
| physics | `physics_quantity`, `design_method` |
| lattice | `lattice_type`, `lattice_component`, `injection_method` |
| magnets | `magnet_type` |
| machines | `reference_machine` |
| software | `software_api` |

This pre-partitioning guarantees that physics quantities never cluster with
software APIs and that reference machines never mix with lattice types. These
are hard boundaries that would be difficult to learn reliably from sparse
TF-IDF similarity alone.

### 4.2 Sub-clustering within partitions

Within each partition, the number of sub-clusters is determined by:

- `1` for `<= 3` subjects
- `2` for `<= 8` subjects
- `3` for `<= 15` subjects
- `min(4, floor(sqrt(N)))` otherwise

Each partition is then clustered using the same sparse TF-IDF K-means as
before. Singleton sub-clusters (with `<= 1` member) are merged into the
nearest same-partition neighbor to avoid fragmentation.

### 4.3 Theme-level clustering

Within each learned domain, a second fixed-count clustering pass is applied to
produce intermediate themes. The number of themes is determined only by the
number of subjects in that domain:

- `1` theme for `<= 2` subjects
- `2` themes for `<= 5` subjects
- `3` themes for `<= 10` subjects
- `4` themes otherwise

### 4.4 Automatic cluster naming

Domain labels are constructed from each partition's base label (e.g.
"Beam Physics", "Lattice Design", "Reference Machines") followed by
representative subject labels as qualifiers. For example:

- `Beam Physics: emittance, circumference, momentum compaction`
- `Lattice Design: Classical Cells (H7BA, 7BA, 9BA)`
- `Reference Machines: Flagships (ESRF-EBS, APS-U, HEPS)`
- `JuTrack API: Optics & Analysis (KQUAD, Lattice, DRIFT, twissring)`

Theme labels within a domain use differential TF-IDF naming: the centroid
vector of each theme is penalized by the sibling themes' centroids so that the
most distinctive tokens are selected for the label. This prevents all themes
in the same domain from receiving the same generic label.

## 5. Plot Layout Method

The plot stored in
[knowledge/.graph/cluster_index.json](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/cluster_index.json)
is a rootless radial layout.

### 5.1 Structural levels

The visible rings encode four semantic levels:

- Domains
- Themes
- Topics (subjects)
- Details (fact leaves)

The plot deliberately omits a synthetic central node in the public view.

### 5.2 Ordering and color continuity

Domains are arranged in a circular order obtained from pairwise cosine
similarity between domain profiles. This gives the plot a smooth color
transition: nearby colors correspond to semantically similar learned regions.

### 5.3 Density management

To prevent visually dominant regions from collapsing into a single congested
arc, the plot allocates angular space according to both cluster weight and
minimum span constraints. Themes and topics are additionally staggered across
nearby radii, so dense local neighborhoods do not sit on exactly one circle.

### 5.4 Label rendering

The SVG renderer wraps labels into short multi-line spans, assigns label
priority by structural importance, and suppresses lower-priority labels when
their bounding boxes overlap already-placed labels. The display is therefore
intentionally selective: omitted labels indicate collision management, not
missing knowledge.

## 6. Release-Safe Transformation

The public repository is not meant to redistribute external source materials.
Accordingly, the current release-safe build keeps only:

- `cluster_index.json`
- `subject_index.json`
- `release_prompt_context.md`
- `stats.json`
- manifest/taxonomy/ontology snapshots

and removes:

- source inventories
- section/chunk-level text
- raw fact dumps
- full provenance-heavy graph exports
- the external materials themselves

Therefore, the public graph is a distilled knowledge representation rather than
a source-archival corpus.

## 7. Current Public Results

The current public bundle reports the following release-safe statistics from
[knowledge/.graph/stats.json](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/stats.json):

| Quantity | Value |
|---|---:|
| Distilled subjects | 69 |
| Learned domains | 10 |
| Learned themes | 25 |
| Detail leaves | 360 |
| Plot nodes | 379 |

The public `source_count` and `chunk_count` are both `0` by design because
source-level provenance has been removed from the released bundle. These zeros
should not be interpreted as indicating that no corpus processing occurred.

### 7.1 Subject-category composition

The current sanitized subject index contains:

| Subject category | Subjects | Linked details |
|---|---:|---:|
| `software_api` | 24 | 111 |
| `reference_machine` | 15 | 137 |
| `physics_quantity` | 13 | 134 |
| `lattice_type` | 12 | 27 |
| `magnet_type` | 3 | 7 |
| `injection_method` | 1 | 1 |
| `lattice_component` | 1 | 1 |

This distribution shows that the current graph is simultaneously carrying
accelerator-physics concepts, reference-machine benchmarks, and JuTrack API
knowledge.

### 7.2 Largest learned domains

The most information-dense public domains at the current release are:

| Learned domain | Subjects | Details |
|---|---:|---:|
| JuTrack API: Optics & Analysis (KQUAD, Lattice, DRIFT, twissring) | 24 | 111 |
| Beam Physics: emittance, circumference, momentum compaction | 11 | 99 |
| Reference Machines: Compact DLSRs (ALS-U, HALF, SKIF) | 7 | 63 |
| Reference Machines: Flagships (ESRF-EBS, APS-U, HEPS) | 3 | 43 |
| Beam Physics: energy, energy loss per turn | 2 | 35 |
| Reference Machines: Flagships (MAX IV, NSLS-II, PETRA IV) | 5 | 31 |

These results show clear separation of physics, lattice design, reference
machines, magnets, and software concerns. Each domain is either purely composed
of a single ontology category or mixes closely related categories within the
same partition (e.g. `lattice_type` and `injection_method`).

## 8. Interpretation and Limitations

The present method has several strengths:

- it is transparent and fully local
- it produces typed facts rather than only embeddings
- it supports release-safe publication
- it creates a deeper learned structure than a static hand-written branch tree

At the same time, several limitations remain:

- extraction is heuristic and ontology-dependent rather than fully semantic
- numerical consistency across facts is not yet validated by physics models
- cluster naming is descriptive and may not always yield ideal terminology
- the public bundle no longer exposes enough provenance for source-level audit

Accordingly, the current graph should be interpreted as a structured research
aid and retrieval substrate, not as a formally verified accelerator handbook.

## 9. Practical Use in This Project

The distilled graph is used in two main ways:

1. The agent can retrieve high-level domain, theme, and topic summaries from the
   stored cluster representation.
2. The code-writing and research prompts can consume the release-safe prompt
   context derived from the graph even after raw source files have been removed.

This makes the knowledge system useful both during internal development and in a
public release setting.
