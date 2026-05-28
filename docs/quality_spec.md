# Quality Specification

## Data

### Ground Truth

Hand-labeled 100 triplets.

| Validation | Test |
| ---------- | ---- |
| 70         | 30   |

## Conversation

- Iterations N = 5
- Samples S = 10
  - samples given to the LLM

## Metrics

- Precision
- Recall

### Generalization

- generalization gap
  - compute the difference between the accuracy on the validation and test set
- schema coverage
  - how many new type of relation appear in the test set