# RecArena Model Catalog

Complete reference of all models available in RecArena with their identifiers, papers, and usage.

---

## Sequential Models

Models for sequential recommendation that learn from ordered user interactions.

### SASRec
- **Model ID**: `sasrec`
- **Full Name**: Self-Attentive Sequential Recommendation
- **Year**: 2018
- **Paper**: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- **Conference**: KDD 2018
- **Architecture**: Transformer with causal self-attention
- **Key Features**:
  - Multi-head self-attention
  - Position embeddings
  - Causal masking for autoregressive prediction
  - Layer normalization and dropout
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_heads`: Number of attention heads
  - `num_layers`: Number of transformer layers
  - `feedforward_dim`: Feedforward network dimension
  - `dropout_rate`: Dropout probability
- **Supported Losses**: `cross_entropy`, `bpr`, `sampled_softmax`, `bce`, `gbce`
- **Usage**:
  ```python
  from rec_arena.models import SASRec
  from rec_arena.configs.defaults.sasrec import SASRecConfig
  
  config = SASRecConfig(vocab_size=1000, embedding_dim=64, num_layers=2)
  model = SASRec(config)
  ```

---

### BERT4Rec
- **Model ID**: `bert4rec`
- **Full Name**: BERT for Sequential Recommendation
- **Year**: 2019
- **Paper**: [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)
- **Conference**: CIKM 2019
- **Architecture**: Bidirectional Transformer with masked language modeling
- **Key Features**:
  - Bidirectional self-attention (no causal masking)
  - Masked language modeling (MLM) training
  - Cloze task for item prediction
  - Configurable masking strategy
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_heads`: Number of attention heads
  - `num_layers`: Number of transformer layers
  - `mask_prob`: Probability of masking tokens (default: 0.15)
  - `mask_token_prob`: Probability of [MASK] replacement (default: 0.8)
  - `random_token_prob`: Probability of random replacement (default: 0.1)
- **Supported Losses**: `mlm`, `cross_entropy`, `sampled_softmax`
- **Usage**:
  ```python
  from rec_arena.models import BERT4Rec
  from rec_arena.configs.defaults.bert4rec import BERT4RecConfig
  
  config = BERT4RecConfig(vocab_size=1000, embedding_dim=64, loss_type="mlm")
  model = BERT4Rec(config)
  ```

---

### GRU4Rec
- **Model ID**: `gru4rec`
- **Full Name**: GRU for Recommendation
- **Year**: 2016
- **Paper**: [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- **Conference**: ICLR 2016
- **Architecture**: Gated Recurrent Unit (GRU) based RNN
- **Key Features**:
  - GRU layers for sequential modeling
  - Efficient for long sequences
  - Packed sequences for variable lengths
  - Dropout regularization
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `hidden_size`: GRU hidden state size
  - `num_layers`: Number of GRU layers
  - `dropout_rate`: Dropout probability
- **Supported Losses**: `cross_entropy`, `bpr`, `sampled_softmax`, `bce`
- **Usage**:
  ```python
  from rec_arena.models import GRU4Rec
  from rec_arena.configs.defaults.gru4rec import GRU4RecConfig
  
  config = GRU4RecConfig(vocab_size=1000, hidden_size=64, num_layers=1)
  model = GRU4Rec(config)
  ```


### Caser
- **Model ID**: `caser`
- **Full Name**: Convolutional Sequence Embedding Recommendation
- **Year**: 2018
- **Paper**: [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://arxiv.org/abs/1809.07426)
- **Conference**: WSDM 2018
- **Architecture**: CNN with horizontal and vertical convolutions
- **Key Features**:
  - Horizontal convolutions for union-level patterns (n-grams)
  - Vertical convolutions for point-level patterns
  - Multiple filter sizes
  - Configurable activation functions
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_horizontal_filters`: Number of horizontal filters
  - `num_vertical_filters`: Number of vertical filters
  - `horizontal_filter_sizes`: List of filter sizes (e.g., [2, 3, 4])
  - `vertical_filter_size`: Vertical filter size
  - `activation`: Activation function (relu, gelu, swish, tanh)
- **Supported Losses**: `cross_entropy`, `bpr`, `bce`
- **Usage**:
  ```python
  from rec_arena.models import Caser
  from rec_arena.configs.defaults.caser import CaserConfig
  
  config = CaserConfig(
      vocab_size=1000, 
      horizontal_filter_sizes=[2, 3, 4],
      activation="gelu"
  )
  model = Caser(config)
  ```

---

### FMLPRec
- **Model ID**: `fmlprec`
- **Full Name**: Filter-enhanced MLP for Sequential Recommendation
- **Year**: 2022
- **Paper**: [Filter-enhanced MLP is All You Need for Sequential Recommendation](https://arxiv.org/abs/2202.13556)
- **Conference**: WWW 2022
- **Architecture**: MLP-only with learnable frequency filters
- **Key Features**:
  - MLP-only architecture (no attention/convolution)
  - Learnable complex filters in frequency domain
  - FFT/IFFT for efficient filtering
  - Tunable MLP hidden dimensions
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_blocks`: Number of FMLP blocks
  - `mlp_hidden_dim`: MLP hidden dimension (default: 4 * embedding_dim)
  - `dropout_rate`: Dropout probability
- **Supported Losses**: `cross_entropy`, `bpr`, `sampled_softmax`
- **Usage**:
  ```python
  from rec_arena.models import FMLPRec
  from rec_arena.configs.defaults.fmlprec import FMLPRecConfig
  
  config = FMLPRecConfig(vocab_size=1000, embedding_dim=64, mlp_hidden_dim=256)
  model = FMLPRec(config)
  ```

---

### RecM
- **Model ID**: `recm`
- **Full Name**: Ensemble Modeling for Sequential Recommendation
- **Year**: 2024
- **Paper**: Custom ensemble architecture
- **Architecture**: Batch ensemble with multiple loss functions
- **Key Features**:
  - Batch ensembling for diversity
  - Multiple loss functions per ensemble member
  - Shared transformer backbone
  - Ensemble-specific projections
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_heads`: Number of attention heads
  - `num_layers`: Number of transformer layers
  - `ensemble_size`: Number of ensemble members
  - `ensemble_loss_functions`: List of loss functions for ensemble
- **Supported Losses**: Multiple (configured per ensemble member)
- **Usage**:
  ```python
  from rec_arena.models import RecM
  from rec_arena.configs.defaults.recm import RecMConfig
  
  config = RecMConfig(
      vocab_size=1000, 
      embedding_dim=64,
      ensemble_size=4,
      ensemble_loss_functions=["cross_entropy", "bpr", "sampled_softmax", "bce"]
  )
  model = RecM(config)
  ```

---



### HSTU
- **Model ID**: `hstu`
- **Full Name**: Hierarchical Sequence Transformer for User Modeling
- **Year**: 2024
- **Paper**: [HSTU: Hierarchical Sequence Transformer](https://arxiv.org/pdf/2402.17152)
- **Conference**: Preprint 2024
- **Architecture**: Hierarchical transformer with item-level and session-level attention
- **Key Features**:
  - Multi-scale pattern capture (item + session level)
  - Session boundary detection
  - Better long-term dependency modeling
  - Configurable session aggregation
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_heads`: Number of attention heads
  - `num_item_layers`: Number of item-level transformer layers
  - `num_session_layers`: Number of session-level transformer layers
  - `session_size`: Number of items per session
  - `session_pooling`: Session aggregation method (mean, max, last)
- **Supported Losses**: `cross_entropy`, `bpr`, `sampled_softmax`, `bce`, `gbce`
- **Usage**:
  ```python
  from rec_arena.models import HSTU
  from rec_arena.configs.defaults.hstu import HSTUConfig
  
  config = HSTUConfig(
      vocab_size=1000, 
      embedding_dim=64,
      num_item_layers=2,
      num_session_layers=1,
      session_size=10
  )
  model = HSTU(config)
  ```

---

### FuXi-α
- **Model ID**: `fuxi`
- **Full Name**: FuXi-α: Adaptive Multi-Channel Attention for Sequential Recommendation
- **Year**: 2024
- **Paper**: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)
- **Conference**: Preprint 2024
- **Architecture**: Adaptive multi-channel attention with separated relative bucketed time and position bias
- **Key Features**:
  - Adaptive multi-channel attention mechanism
  - Separated relative bucketed time and position bias
  - SiLU-gated linear projections
  - Multi-stage FFN with RMSNorm
  - Configurable single-stage FFN mode
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_heads`: Number of attention heads
  - `num_layers`: Number of transformer layers
  - `attention_dim`: Attention projection dimension
  - `linear_dim`: Linear projection dimension
  - `dropout_rate`: Dropout probability
  - `ffn_multiply`: FFN dimension multiplier
  - `ffn_single_stage`: Use single-stage FFN instead of multi-stage
- **Supported Losses**: `cross_entropy`, `bpr`, `sampled_softmax`, `bce`, `gbce`
- **Usage**:
  ```python
  from rec_arena.models import FuXi
  from rec_arena.configs.defaults.fuxi import FuXiConfig
  
  config = FuXiConfig(
      vocab_size=1000,
      embedding_dim=64,
      num_heads=2,
      num_layers=2,
      attention_dim=64
  )
  model = FuXi(config)
  ```

---

### FuXi-γ
- **Model ID**: `fuxi_gamma`
- **Full Name**: FuXi-γ: Exponential-Power Temporal Encoding for Sequential Recommendation
- **Year**: 2024
- **Paper**: [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)
- **Conference**: Preprint 2024
- **Architecture**: Exponential-power temporal encoding with SwiGLU FFN
- **Key Features**:
  - Exponential-power temporal encoding with learnable α/β/γ parameters
  - SwiGLU feed-forward network
  - RMSNorm normalization
  - Configurable temporal decay ranges
  - Learnable or fixed gamma parameters
- **Tunable Parameters**:
  - `embedding_dim`: Embedding dimension
  - `num_heads`: Number of attention heads
  - `num_layers`: Number of transformer layers
  - `attention_dim`: Attention projection dimension
  - `linear_dim`: Linear projection dimension
  - `dropout_rate`: Dropout probability
  - `ffn_multiply`: FFN dimension multiplier
  - `range_alpha`: Alpha range for temporal encoding
  - `left_beta` / `right_beta`: Beta range bounds
  - `gamma_learnable`: Whether gamma parameters are learnable
- **Supported Losses**: `cross_entropy`, `bpr`, `sampled_softmax`, `bce`, `gbce`
- **Usage**:
  ```python
  from rec_arena.models import FuXiGamma
  from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
  
  config = FuXiGammaConfig(
      vocab_size=1000,
      embedding_dim=64,
      num_heads=2,
      num_layers=2,
      range_alpha=0.1
  )
  model = FuXiGamma(config)
  ```

---

## Implicit Feedback Models

Models for user-item interaction prediction without sequential order.

### NCF
- **Model ID**: `ncf`
- **Full Name**: Neural Collaborative Filtering
- **Year**: 2017
- **Paper**: [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- **Conference**: WWW 2017
- **Architecture**: GMF + MLP fusion
- **Key Features**:
  - Generalized Matrix Factorization (GMF) component
  - Multi-Layer Perceptron (MLP) component
  - Fusion of GMF and MLP outputs
  - Configurable MLP architecture
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `embedding_dim`: Embedding dimension for GMF
  - `hidden_dims`: List of MLP hidden dimensions
  - `dropout_rate`: Dropout probability
  - `activation`: Activation function
- **Supported Losses**: `bce`, `bpr`
- **Usage**:
  ```python
  from rec_arena.models import NCF
  from rec_arena.configs.defaults.ncf import NCFConfig
  
  config = NCFConfig(
      num_users=1000, 
      num_items=500, 
      embedding_dim=64,
      hidden_dims=[128, 64, 32]
  )
  model = NCF(config)
  ```

---

### TwoTower
- **Model ID**: `twotower`
- **Full Name**: Two-Tower Dual Encoder
- **Year**: 2019
- **Paper**: [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/pub48840/)
- **Conference**: RecSys 2019
- **Architecture**: Dual encoder with separate user/item towers
- **Key Features**:
  - Separate user and item encoders
  - Shared embedding space
  - Efficient for large-scale retrieval
  - Configurable tower architectures
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `embedding_dim`: Embedding dimension
  - `user_tower_dims`: List of user tower hidden dimensions
  - `item_tower_dims`: List of item tower hidden dimensions
  - `activation`: Activation function
- **Supported Losses**: `bce`, `bpr`
- **Usage**:
  ```python
  from rec_arena.models import TwoTower
  from rec_arena.configs.defaults.twotower import TwoTowerConfig
  
  config = TwoTowerConfig(
      num_users=1000, 
      num_items=500,
      user_tower_dims=[128, 64],
      item_tower_dims=[128, 64]
  )
  model = TwoTower(config)
  ```

---

### SimpleX
- **Model ID**: `simplex`
- **Full Name**: SimpleX - A Simple and Strong Baseline
- **Year**: 2021
- **Paper**: [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://arxiv.org/abs/2109.12613)
- **Conference**: CIKM 2021
- **Architecture**: Cosine similarity with normalized embeddings
- **Key Features**:
  - Cosine similarity for user-item matching
  - Normalized embeddings
  - History aggregation (mean/sum/max)
  - Simple yet effective
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `embedding_dim`: Embedding dimension
  - `history_aggregation`: Aggregation method (mean, sum, max)
- **Supported Losses**: `bce`, `bpr`
- **Usage**:
  ```python
  from rec_arena.models import SimpleX
  from rec_arena.configs.defaults.simplex import SimpleXConfig
  
  config = SimpleXConfig(
      num_users=1000, 
      num_items=500,
      embedding_dim=64,
      history_aggregation="mean"
  )
  model = SimpleX(config)
  ```

---

### BPR-MF
- **Model ID**: `bprmf`
- **Full Name**: Bayesian Personalized Ranking Matrix Factorization
- **Year**: 2009
- **Paper**: [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- **Conference**: UAI 2009
- **Architecture**: Matrix factorization with pairwise ranking
- **Key Features**:
  - Classic matrix factorization
  - Pairwise ranking optimization
  - Simple and interpretable
  - Efficient for large-scale data
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `embedding_dim`: Embedding dimension
  - `init_std`: Initialization standard deviation
- **Supported Losses**: `bpr`, `bce`
- **Usage**:
  ```python
  from rec_arena.models import BPRMF
  from rec_arena.configs.defaults.bprmf import BPRMFConfig
  
  config = BPRMFConfig(num_users=1000, num_items=500, embedding_dim=64)
  model = BPRMF(config)
  ```

---

### EASE
- **Model ID**: `ease`
- **Full Name**: Embarrassingly Shallow Autoencoders for Sparse Data
- **Year**: 2019
- **Paper**: [EASE: Embarrassingly Shallow Autoencoders](https://dl.acm.org/doi/10.1145/3308558.3313710)
- **Conference**: WWW 2019
- **Architecture**: Linear autoencoder with closed-form solution
- **Key Features**:
  - Closed-form solution (no training iterations!)
  - Extremely fast training (seconds)
  - Often beats complex neural models
  - Item-item similarity matrix
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `reg_lambda`: Regularization parameter (typical: 100-1000)
- **Supported Losses**: N/A (closed-form)
- **Usage**:
  ```python
  from rec_arena.models import EASE
  from rec_arena.configs.defaults.ease import EASEConfig
  
  config = EASEConfig(num_users=1000, num_items=500, reg_lambda=500)
  model = EASE(config)
  model.fit(train_data)  # No iterations!
  ```

---

### SLIM
- **Model ID**: `slim`
- **Full Name**: Sparse Linear Methods for Top-N Recommender Systems
- **Year**: 2011
- **Paper**: [SLIM: Sparse Linear Methods](https://ieeexplore.ieee.org/document/6137254)
- **Conference**: ICDM 2011
- **Architecture**: Sparse item-item similarity with elastic net
- **Key Features**:
  - Sparse linear model
  - L1 + L2 regularization (elastic net)
  - Interpretable item similarities
  - Strong performance on sparse data
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `alpha`: Regularization strength
  - `l1_ratio`: L1 vs L2 ratio (0=L2 only, 1=L1 only)
- **Supported Losses**: N/A (elastic net)
- **Usage**:
  ```python
  from rec_arena.models import SLIM
  from rec_arena.configs.defaults.slim import SLIMConfig
  
  config = SLIMConfig(num_users=1000, num_items=500, alpha=0.1, l1_ratio=0.1)
  model = SLIM(config)
  model.fit(train_data)
  ```

---

### ItemKNN
- **Model ID**: `itemknn`
- **Full Name**: Item-based K-Nearest Neighbors Collaborative Filtering
- **Year**: 2001
- **Paper**: [Item-based Collaborative Filtering](https://dl.acm.org/doi/10.1145/371920.372071)
- **Conference**: WWW 2001
- **Architecture**: Item-based collaborative filtering with cosine similarity
- **Key Features**:
  - Classic baseline method
  - Simple and interpretable
  - Fast inference
  - Works well with sparse data
- **Tunable Parameters**:
  - `num_users`: Number of users
  - `num_items`: Number of items
  - `k`: Number of neighbors
  - `similarity`: Similarity metric ('cosine' or 'jaccard')
  - `shrinkage`: Shrinkage parameter for cosine similarity
  - `normalize`: Normalize similarity scores
- **Supported Losses**: N/A (similarity-based)
- **Usage**:
  ```python
  from rec_arena.models import ItemKNN
  from rec_arena.configs.defaults.itemknn import ItemKNNConfig
  
  config = ItemKNNConfig(num_users=1000, num_items=500, k=100, similarity="cosine")
  model = ItemKNN(config)
  model.fit(train_data)
  ```

---

## Model Selection Guide

### When to Use Sequential Models

Use sequential models when:
- You have ordered user interactions (browsing history, purchase sequences)
- Temporal patterns are important
- You want to predict the next item in a sequence

**Recommended Models**:
- **SASRec**: Best overall performance, good for most cases
- **BERT4Rec**: When bidirectional context helps (e.g., filling gaps)
- **GRU4Rec**: When you need efficiency and simplicity
- **Caser**: When you want to capture local patterns (n-grams)
- **FMLPRec**: When you want MLP-only architecture

### When to Use Implicit Models

Use implicit models when:
- You have user-item interactions without order
- You want to predict user-item affinity
- You need efficient large-scale retrieval

**Recommended Models**:
- **NCF**: Best overall performance for implicit feedback
- **TwoTower**: When you need efficient retrieval at scale
- **SimpleX**: When you want simplicity and interpretability
- **BPR-MF**: When you need a simple baseline

---

## Quick Reference Table

| Model | Type | Architecture | Complexity | Best For |
|-------|------|--------------|------------|----------|
| SASRec | Sequential | Transformer | O(n²) | General sequential |
| BERT4Rec | Sequential | Transformer | O(n²) | Bidirectional context |
| GRU4Rec | Sequential | RNN | O(n) | Long sequences |
| Caser | Sequential | CNN | O(n) | Local patterns |
| FMLPRec | Sequential | MLP | O(n) | MLP-only |
| RecM | Sequential | Ensemble | O(n²) | Ensemble diversity |
| HSTU | Sequential | Hierarchical | O(n²) | Multi-scale patterns |
| FuXi-α | Sequential | Adaptive attention | O(n²) | Feature-interaction modeling |
| FuXi-γ | Sequential | Temporal attention | O(n²) | Temporal-aware modeling |
| NCF | Implicit | GMF+MLP | O(1) | General implicit |
| TwoTower | Implicit | Dual encoder | O(1) | Large-scale retrieval |
| SimpleX | Implicit | Cosine | O(1) | Simple baseline |
| BPR-MF | Implicit | MF | O(1) | Classic baseline |
| EASE | Traditional | Linear autoencoder | O(1) | Strong baseline |
| SLIM | Traditional | Sparse linear | O(1) | Interpretable |
| ItemKNN | Traditional | Similarity-based | O(1) | Classic baseline |

---

## Advanced Features

### Semantic IDs

RecArena supports semantic ID tokenization for improved generalization and cold-start performance.

**What are Semantic IDs?**
- Instead of random item IDs, items with similar features get similar IDs
- Improves generalization to unseen items
- Better cold-start performance
- Reduces effective vocabulary size

**Methods**:
- **K-Means Clustering**: Group similar items based on features
- **VQ-VAE**: Learn discrete codes via vector quantization
- **Hierarchical**: Multi-level semantic codes

**Usage**:
```python
from rec_arena.models import SASRec
from rec_arena.tokenizers import SemanticIDModel
import numpy as np

# Create base model
base_model = SASRec(config)

# Wrap with semantic IDs
semantic_model = SemanticIDModel(
    base_model=base_model,
    num_items=1000,
    num_codes=512,  # Reduce vocabulary
    method="kmeans"
)

# Fit semantic IDs from item features
item_features = np.random.randn(1000, 64)  # Use real features
semantic_model.fit_semantic_ids(item_features)

# Train normally
trainer.fit(semantic_model, datamodule)
```

**Compatible with all sequential models**: SASRec, BERT4Rec, GRU4Rec, LiGR, HSTU, etc.

---

## Citation

When using these models, please cite the original papers. See individual model sections for paper links.
