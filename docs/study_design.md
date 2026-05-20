# Study Design

## Changelog
### v1 - 21/05/2026
#### Changes
expanded document
#### Comment
refined evaluation process

## Baseline/Control condition

KG extraction without LLMs

## Evaluation

### Downstream

person-relation schema

---

On a small set of hand-labeled triplets

|                 | known data     | unknown data   |
| --------------- | -------------- | -------------- |
| one-shot prompt | conversational | generalization |
| conversational  | conversational | generalization |

#### Conversational VS one-shot

On "validation" data: compare the resulting graph from a conversation and a one-shot prompt

#### Generalization

On test set: compare how the system generalize