# Knowledge Corpus

This directory now supports two modes:

- Private build mode: maintainers can place local notes, papers, notebooks, and API references here to refresh the knowledge graph.
- Public release mode: the repository keeps only distilled knowledge artifacts and removes bundled third-party source materials.

## Build the distilled graph

Run:

```bash
python scripts/build_knowledge_graph.py
```

Outputs are written to [knowledge/.graph](/c:/Users/86186/Desktop/accdesign/knowledge/.graph).

For a formal description of the extraction, clustering, release-safe
transformation, and current public results, see
[METHODOLOGY.md](/c:/Users/86186/Desktop/accdesign/knowledge/METHODOLOGY.md).

## Public release artifacts

The release-safe bundle keeps only distilled, reusable knowledge:

- [manifest.json](/c:/Users/86186/Desktop/accdesign/knowledge/manifest.json): build configuration
- [taxonomy.json](/c:/Users/86186/Desktop/accdesign/knowledge/taxonomy.json): controlled vocabulary and aliases
- [ontology.json](/c:/Users/86186/Desktop/accdesign/knowledge/ontology.json): relation and quantity rules
- [cluster_index.json](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/cluster_index.json): multi-level semantic cluster graph plus the stored plot layout used by the UI
- [subject_index.json](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/subject_index.json): sanitized subject-level index
- [release_prompt_context.md](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/release_prompt_context.md): compact prompt context derived from the graph
- [stats.json](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/stats.json): release-safe graph statistics

## Current clustering plot

The current visualization is a rootless radial semantic map. It does not place
an invented central node such as "Accelerator Design Knowledge" in the visible
plot. Instead, the inner structure starts directly from learned domain clusters
derived from the subject graph.

The rings are interpreted as:

- `Domains`: the broadest learned semantic clusters
- `Themes`: secondary groups formed inside each domain
- `Topics`: reusable subject concepts such as machine names, lattice types, APIs, or physics quantities
- `Details`: outer factual leaves connected to each topic

The layout is designed to reflect knowledge structure rather than file
organization:

- Domains are ordered by semantic similarity, so neighboring colors transition smoothly around the map.
- Angular span is weighted by cluster importance and subject density, so larger knowledge regions receive more room.
- Themes and topics are staggered across nearby radii instead of sitting on one exact circle, which reduces local crowding.
- Labels are wrapped and overlap-filtered in the SVG renderer, so not every possible label is shown at once.
- The focus selector in the UI dims unrelated domains and makes the chosen region easier to inspect.

In practice, this means the plot is intentionally selective. The graph stores
more nodes than the renderer labels at once, and label suppression is treated as
a readability feature, not as missing knowledge.

## Stored plot data

Inside [cluster_index.json](/c:/Users/86186/Desktop/accdesign/knowledge/.graph/cluster_index.json), the `plot`
section contains:

- `rings`: the semantic ring names and radii
- `nodes`: positioned domain/theme/topic/detail nodes
- `edges`: radial hierarchy edges plus similarity bridges between neighboring domains
- `view_box`: the SVG framing used by the UI

This means the public bundle keeps a reusable plot layout without retaining the
private raw-source corpus.

## What is intentionally removed in release mode

These artifacts are treated as private build intermediates and should not ship in the public bundle:

- raw source files such as PDFs, notebooks, and private notes
- `sources.json`, `sections.json`
- `chunks.jsonl`, `facts.jsonl`
- `nodes.jsonl`, `edges.jsonl`
- `knowledge_graph.json`
- `quality_report.json`

## Adding new material later

When maintainers want to refresh the knowledge privately:

1. Add the new local materials under `knowledge/`.
2. Update [manifest.json](/c:/Users/86186/Desktop/accdesign/knowledge/manifest.json) if new discovery rules or source profiles are needed.
3. Update [taxonomy.json](/c:/Users/86186/Desktop/accdesign/knowledge/taxonomy.json) or [ontology.json](/c:/Users/86186/Desktop/accdesign/knowledge/ontology.json) if the new material introduces new concept families or structured quantities.
4. Rebuild with `python scripts/build_knowledge_graph.py`.
5. Before publishing, switch back to the release-safe bundle and remove the external source files.

## Design intent

The released project should look like a knowledge system, not a packaged copy of external references. The retained graph is therefore:

- clustered by learned semantic structure rather than by file names
- sanitized to remove direct source provenance
- still rich enough to support retrieval, prompting, and visualization
