RoPE-Transformer LLM Training


A compact transformer language model trained from scratch using RoPE (Rotary Position Embeddings) on WikiText-103 dataset. Built with PyTorch.

Model Architecture


Parameter	Value	Description
embedding_dim	256	Token embedding size
ff_dim	1024	Feed-forward dimension (4× embedding)
num_heads	8	Multi-head attention
num_layers	6	Transformer layers
dropout	0.1	Regularization
Total Parameters: ~8M (research-friendly size)

Dataset
Source: Salesforce/wikitext (wikitext-103-v1)

Samples: 50K shuffled rows for efficient training

Task: Causal language modeling

RoPE: Rotary Position Embeddings


Traditional Positional Encoding Limitations
text
Standard approach: x_pos = x_token + PE(position)
Problems:

Absolute position dependency

Extrapolation failure beyond training lengths

Fixed position bias

RoPE Core Innovation
RoPE transforms positional information into rotation matrices applied directly to query/key vectors:

text
For dimension pair (x_{2i}, x_{2i+1}) at position m:
[x_real, x_imag] → R(θ_m) × [x_real, x_imag]

Where: θ_m,i = m × 10000^(-2i/d)
R(θ) = [[cosθ, -sinθ],
        [sinθ,  cosθ]]
Key Mathematical Property:

text


Q_mᵀK_n = f(m-n)  # Relative position only!
No absolute position dependency
Implementation Process
text
1. Precompute frequencies: θ_i = 10000^(-2i/d)
2. Generate cos/sin: [cos(θ×pos), sin(θ×pos)] ∀ positions
3. Split Q/K into 2D pairs along head_dim
4. Rotate each pair: [x₁cosθ-x₂sinθ, x₁sinθ+x₂cosθ]
5. Attention computation preserves relative distances
Shape transformations:

text

Q ∈ ℝᴮᴴᵀᴰ → Split → [ℝᴮᴴᵀᴰᐟ² × 2] → Rotate → ℝᴮᴴᵀᴰ
          ↑
    Apply identical θ to both dimensions of each pair
 Model Architecture Flow
text
Input Tokens → Embedding → [TransformerBlock × 6] → LM Head
                      ↓
              Positional Encoding: NONE (RoPE handles internally)
Single TransformerBlock:

text


Input → LayerNorm → RoPE-MultiHeadAttention → Residual
        ↓
     LayerNorm → FeedForward(256→1024→256) → Residual
🔬 RoPE Advantages (Theoretical)
Relative Positioning: Attention scores depend only on position_i - position_j

Length Extrapolation: Works beyond training sequence lengths

Data Efficiency: No positional embedding parameters

Stability: Bounded rotation matrices prevent exploding gradients

📈 Training Dynamics


Loss Progression (expected):

text
Epoch 1: High loss (learning token dependencies)
Epoch 5: Grammar acquisition
Epoch 10: Perplexity ~30 (reasonable for 8M model)
Optimization: Causal language modeling with cross-entropy loss on next-token prediction.

💾 Model Persistence


State Dictionary Format:

text
model.state_dict() = {
    'query_projection.weight': (256, 256),
    'cached_position_cosines': (max_seq_len, 1, 256),  # RoPE cache
    ...
}
RoPE Caching: Precomputed cos/sin matrices regenerate automatically on first forward pass after loading.
