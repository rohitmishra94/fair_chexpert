# fair_chexpert
fairness architecture R&amp;D and experiment

## data information
https://aimi.stanford.edu/datasets/chexpert-chest-x-rays

### Table summary for v2
| Component               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| Feature Extraction      | Pre-trained ResNet50, adaptive pooling to 2048-dimensional vector.          |
| Disease Branch          | Encoder (2048→128), head (128→1 for binary classification), frozen decoder. |
| Race Branch             | Encoder (2048→128), head (128→6 for multi-class), frozen decoder.           |
| Disease Loss            | Binary cross-entropy with logits.                                           |
| Race Loss               | Cross-entropy for multi-class classification.                               |
| Orthogonality Loss      | Mean absolute cosine similarity of decoded features, encourages orthogonality. |
| Constraint Loss         | Mean absolute difference of squared norms, ensures norm preservation.       |
| Training Strategy       | Three separate forward passes for disease, race, and constraint updates.    |

