# Final Report

## Research Questions
- Does conversational graph extraction produce better results than using a predefined ontology?
- Does the schema generalize or overfit the conversation?

## Positioning & Related Work
Current research heavily emphasizes either fully automated LLM knowledge graph construction (e.g., AutoSchemaKG) or expert-in-the-loop systems (e.g., SCHEMA-MINER). There is a clear gap in studying KG generation driven by free dialogue from a non-expert user discovering their requirements interactively.

- **Baseline**: Standard NLP extraction via SpaCy.
- **LLM Engine**: [Mistral Small 4](https://mistral.ai/news/mistral-small-4/) through Mistral official APIs


## Study Design
- **Data**: 325 emails from `jeeproject@yahoo.com` (Epstein Files).
- **Data Splits**: Validation Set (228 emails, ~70%), Test Set (97 emails, ~30%).
- **Conditions**: Baseline (NLP), One-Shot Prompt, Conversational (5 turns).
- **Evaluation**: Hand-labeled ground truth triplets evaluated for Precision and Generalization.

## Statistics & Results

### Schema Discovery
Over 5 conversational turns, the schema dynamically adapted to user constraints. `Location` and `Transaction` entities were explicitly added based on feedback, while `Date` and `Quantity` were rejected to limit noise.

### Performance Metrics (Precision)
Bootstrap resampling applied to 50 samples per condition.

| Condition      | Data          | Mean Precision | 95% CI       |
| -------------- | ------------- | -------------- | ------------ |
| Baseline       | Unseen        | 0.54           | [0.42, 0.66] |
| One-Shot       | Seen (Val)    | 1.00           | [1.00, 1.00] |
| One-Shot       | Unseen (Test) | 0.90           | [0.80, 0.98] |
| Conversational | Seen (Val)    | 0.88           | [0.78, 0.96] |
| Conversational | Unseen (Test) | 0.84           | [0.74, 0.94] |

### Generalization & Coverage
- **Generalization Gap**: The one-shot approach degraded by 10% on unseen data (1.00 -> 0.90), while the conversational approach only degraded by 4% (0.88 -> 0.84), showing higher stability and less overfitting.
- **Schema Coverage**: Both LLM approaches achieved a 1.0 schema coverage ratio, successfully utilizing 8 distinct relation types. Although it has to be said that this result depends heavily on the conversation and the person doing the manual labeling.
- **Entity Resolution**: The pipeline achieved a 0% reduction during resolution (760 initial nodes -> 760 final nodes). It has to be noted that the LLM does really grate entity resolution by itself.