# Study Plan

## Changelog
### v0 - 11/05/2026
#### Changes
#### Comment
init

## Research Questions

- Does conversational graph extraction produce better results than using a predefined ontology?

Other possible questions:
- conversation for N turns VS one careful prompt
- when does the schema converge?
- does the schema generalize to unseen part or it overfits the part seen by the user?
- expert guided VS LLM proposed + user approved

## Open Questions

- How to do evaluation?
- How to do evaluation without a ground truth?

One idea is to do downstream evaluation. Judge how much the graph is useful for another task.

Judge how the graph answers to the question: who knows who? who has been where and when?

Ground truth is needed. Maybe build the ground truth on a subset of files and hope that the model can generalize on the whole data.

- LLM as a judge
  - how to measure if it does a good job

- What tools should I use?
- What is BERT, should I use it?

extract data from pdf, data quality