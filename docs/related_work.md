# Related Work

## Changelog
### v0 - 11/05/2026
#### Changes
#### Comment
init

## Papers

### General

- Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction
  - https://arxiv.org/abs/2404.03868
  - methodology on how to create KGs with the help of LLMs
- AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora
  - https://arxiv.org/abs/2505.23628
  - create high quality KG with LLMs without human intervention
  - cons:
    - computationally intensive
    - biasis and limitations of underlying model
- Automated Knowledge Graph Construction using Large Language Models
and Sentence Complexity Modelling
  - https://arxiv.org/pdf/2509.17289v1 
- From human experts to machines: An LLM supported approach to ontology and knowledge graph construction
  - https://arxiv.org/pdf/2403.08345
  - LLMs as copilot to create an ontology
  - cons: output is very sensitive to prompt
- LLM-Driven Ontology Construction for
Enterprise Knowledge Graphs
  - https://arxiv.org/pdf/2602.01276v1
  - focus on enterprise data

### Specific

- LLMs4SchemaDiscovery: A Human-in-the-Loop Workflow for Scientific Schema Mining with Large Language Models
  - AKA SCHEMA-MINER
  - https://arxiv.org/pdf/2504.00752
  - LLM + expert feedback

### Surveys

- LLM-EMPOWERED KNOWLEDGE GRAPH CONSTRUCTION: A SURVEY
  - https://arxiv.org/pdf/2510.20345
- A survey on cutting-edge relation extraction techniques based on language models
  - https://link.springer.com/article/10.1007/s10462-025-11280-0
- A survey on cutting-edge relation extraction techniques based on language models
  - https://link.springer.com/content/pdf/10.1007/s10462-025-11280-0.pdf
- LLMs4OL 2025 Overview: The 2nd Large LanguageModels for Ontology Learning Challenge
  - https://www.tib-op.org/ojs/index.php/ocp/article/view/2913/2922
- https://www.emergentmind.com/topics/interactive-knowledge-graph
  - AI collection of papers on the topic

## Tools

### LLM

still to decide depending on what runs on my hardware

#### Natural Language Processing/Relationship extractor

- https://spacy.io/

This represent the baseline method

### Embedding

needed to compare relationship similarity when comparing resulting schemas

- https://www.sbert.net/

### Database

- https://docs.graphfoundation.org/docs/intro
    - open and free KG database. Disk storage.
    - scalable -> overkill
- https://networkx.org/en/
  - python KG analyzer/explorer
  - RAM storage

### Logging

Python logging module

### Analysis

- numpy
- matplotlib

## Positioning

Looking at papers it seems there is a gap in studying the kg resulting from free dialogue from a non-expert user who has not decided yet what it wants upfront.

Other studies focus on the LLM working automatically or experts being involved.

Assuming a non expert user:
this project focuses in understanding if conversational schema discovery produces a better schema than a one-shot prompt.