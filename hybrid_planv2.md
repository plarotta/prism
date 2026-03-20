# Hybrid PRISM v2: Memory-Augmented Sub-Quadratic Architecture

## Vision

Build a sequence model that matches Transformer quality on reasoning tasks while scaling linearly with sequence length. The core insight: recurrence handles bulk sequence processing in O(n), a fixed-size memory bank enables content-based retrieval in O(n·k), and the total cost remains linear in n.

---

## What We Know So Far

From the PRISM experiments:

- All-slow bidirectional gated linear recurrence with Hadamard projections produces strong sequence embeddings (0.764 nDCG@10 at 5000 steps on LoCoV1, beating geometric decay at 0.692 and Transformers at 0.672)
- The recurrence is a good memory bus — high-retention (λ=0.99) channels with learned input gating create differentiated representations without temporal stratification
- Mean pooling is sufficient for embeddings; attentive pooling provides marginal benefit at 8K (+1.8 nDCG@10) while slightly hurting at 2K (-1.7)
- The architecture scales linearly and operates at sequence lengths where Transformers OOM
- **What it can't do:** content-based retrieval. The input gate is blind to the hidden state. The model cannot condition its processing on what it has previously seen.

This limitation is fatal for general language understanding. Embeddings only need Level 1 (bag of features). Real language requires Level 3+ (content-based retrieval, multi-step reasoning).

---

## The Architecture: PRISM + Memory Bank

### Overview

```
Input tokens
    ↓
[Embedding + Position]
    ↓
[PRISM Block × 4]     ← O(n·d²) linear-time recurrence
    ↓
[Memory Write]         ← compress PRISM states into k memory slots
    ↓
[Memory Read Block]    ← O(n·k·d) cross-attention: tokens query memory
    ↓
[PRISM Block × 4]     ← recurrence continues with attention-refined states
    ↓
[Memory Write]         ← update memory with new information
    ↓
[Memory Read Block]    ← second round of content-based retrieval
    ↓
[PRISM Block × 4]     ← final recurrence layers
    ↓
[Output head]          ← task-dependent: pooling for embeddings, LM head for generation
```

Total complexity: O(L_prism · n · d²) + O(L_mem · n · k · d)

With L_prism=12 recurrence layers, L_mem=2 memory layers, k=32 slots (initial), d=384: the memory layers add roughly 1.5% overhead over pure recurrence. The architecture is dominated by the PRISM blocks and scales linearly in n. k can be scaled to 64 or 128 if tasks demand more memory capacity.

### Component 1: PRISM Blocks (existing)

Use the all-slow configuration validated in Experiment 4:
- 6 channels per layer, all with λ=0.99
- Hadamard-initialized orthogonal subspace projections
- Learned input gating: g_t = σ(W_g · z_t)
- Bidirectional with gated fusion
- Pre-norm residual MLP

No changes needed. This is the proven backbone.

### Component 2: Memory Bank

A fixed-size set of k learned vectors that persist across the sequence. These are the "RAM" of the model — a compressed, content-addressable representation of the sequence.

```python
class MemoryBank(nn.Module):
    def __init__(self, k, d):
        self.slots = nn.Parameter(torch.randn(k, d) * 0.02)  # (k, d)
```

Initialized as learned parameters. Each slot starts as a learned "query template" that specializes during training to capture different types of information (entities, relations, topics, etc.).

### Component 3: Memory Write

After a stack of PRISM layers, compress the token-level hidden states into the memory bank. This is where the recurrence output gets "indexed" into retrievable form.

```python
class MemoryWrite(nn.Module):
    """
    Tokens write to memory slots via cross-attention.
    Each slot attends to all tokens and accumulates a weighted summary.
    
    Complexity: O(n · k · d) — linear in n for fixed k
    """
    def __init__(self, d, k, n_heads=4):
        self.slot_queries = nn.Linear(d, d)    # slots query the sequence
        self.token_keys = nn.Linear(d, d)
        self.token_values = nn.Linear(d, d)
        self.n_heads = n_heads
        self.gate = nn.Linear(2 * d, d)        # gated residual update
    
    def forward(self, memory, token_states, mask):
        # memory: (B, k, d) — current memory state
        # token_states: (B, n, d) — PRISM output
        
        Q = self.slot_queries(memory)           # (B, k, d)
        K = self.token_keys(token_states)       # (B, n, d)
        V = self.token_values(token_states)     # (B, n, d)
        
        # Multi-head cross-attention: slots attend to tokens
        # (B, k, d) × (B, n, d)^T → (B, k, n) → softmax → (B, k, d)
        attn_scores = einsum(Q, K) / sqrt(d)    # (B, k, n)
        attn_scores = masked_fill(attn_scores, ~mask, -inf)
        attn_weights = softmax(attn_scores, dim=-1)
        retrieved = einsum(attn_weights, V)      # (B, k, d)
        
        # Gated update: memory can retain old content or overwrite
        gate = sigmoid(self.gate(cat(memory, retrieved)))
        memory = gate * memory + (1 - gate) * retrieved
        
        return memory
```

The gated update is important — it lets memory slots retain information across multiple write operations. Without it, the second write would overwrite the first. With gating, the model can learn to accumulate across writes.

### Component 4: Memory Read

Tokens query the memory bank to retrieve relevant stored information. This is the content-based retrieval mechanism — the thing that all-slow PRISM cannot do.

```python
class MemoryRead(nn.Module):
    """
    Tokens read from memory slots via cross-attention.
    Each token queries memory and retrieves relevant stored information.
    
    Complexity: O(n · k · d) — linear in n for fixed k
    """
    def __init__(self, d, k, n_heads=4):
        self.token_queries = nn.Linear(d, d)
        self.slot_keys = nn.Linear(d, d)
        self.slot_values = nn.Linear(d, d)
        self.n_heads = n_heads
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
    
    def forward(self, token_states, memory, mask):
        # token_states: (B, n, d)
        # memory: (B, k, d)
        
        residual = token_states
        
        Q = self.token_queries(token_states)    # (B, n, d)
        K = self.slot_keys(memory)              # (B, k, d)
        V = self.slot_values(memory)            # (B, k, d)
        
        # Cross-attention: tokens attend to memory slots
        # (B, n, d) × (B, k, d)^T → (B, n, k) → softmax → (B, n, d)
        attn_scores = einsum(Q, K) / sqrt(d)    # (B, n, k)
        attn_weights = softmax(attn_scores, dim=-1)
        retrieved = einsum(attn_weights, V)      # (B, n, d)
        
        # Residual connection
        output = self.norm(residual + self.out_proj(retrieved))
        
        return output
```

The key: this is O(n·k·d), not O(n²·d). Each token attends to k memory slots (k=32 initially, up to 128), not to all n tokens. The memory bank acts as a bottleneck that forces the model to compress the sequence into a tractable number of retrievable "concepts."

### Why This Might Work Where Cross-Scale Interference Failed

The earlier cross-scale interference tried to mix information between channels using bilinear products. It failed because:
1. The mixing happened at every position regardless of need
2. The bilinear interaction was unstructured — no content-based routing
3. Numerical issues (zero initialization, scale mismatch)

The memory bank approach is fundamentally different:
1. Information routing is content-based — tokens query memory by semantic similarity
2. The memory bottleneck (k << n) forces compression, which acts as regularization
3. Cross-attention is a well-understood, numerically stable mechanism
4. Write and read are separated — the model can learn what to store independently from what to retrieve

---

## Experimental Roadmap

### Phase 1: Proof of Concept on Existing Tasks (1-2 weeks)

**Goal:** Validate that memory augmentation doesn't hurt embedding quality and can help on harder tasks.

#### 1.1 — Implement the Architecture

- Add MemoryBank, MemoryWrite, MemoryRead modules
- Create HybridPRISM that interleaves PRISM blocks with memory operations
- Configuration: 12 PRISM layers split into 3 groups of 4, with memory write/read between groups
- k=32 memory slots initially (conservative, easy to scale)
- Match parameter count to existing models (~20M) by slightly reducing PRISM layer width

#### 1.2 — Sanity Check on LoCoV1

- Train HybridPRISM on LoCoV1 at max_len=2048, same protocol as Experiment 0
- **Expected:** should match or slightly beat all-slow PRISM (0.77+)
- If it degrades: the memory mechanism is interfering with what already works. Debug before proceeding.
- This is a regression test, not the goal. Embeddings don't need memory — this just confirms nothing is broken.

#### 1.3 — Needle in a Haystack Test

Build a synthetic task that specifically requires content-based retrieval:
- Generate documents of length n with a "key" planted at a random position and a "query" at another random position
- The model must match the query to the key based on semantic content, not position
- Example: "The secret code is BLUE" buried in a 4K-token document, with the query "What is the secret code?" at the end
- All-slow PRISM should fail at this beyond short sequences (the key gets averaged away in the recurrence)
- HybridPRISM with memory should succeed (the key gets written to a memory slot and retrieved when the query arrives)

This is the minimum viable demonstration that memory adds Level 3 capability.

#### 1.4 — Scaling Behavior

- Run the needle-in-a-haystack test at lengths 512, 1K, 2K, 4K, 8K, 16K
- Plot accuracy vs length for: all-slow PRISM, HybridPRISM, Transformer (until it OOMs)
- **Success criterion:** HybridPRISM maintains high accuracy at lengths where both all-slow PRISM fails (key forgotten) and Transformer OOMs (too expensive)

### Phase 2: Real Reasoning Tasks (2-4 weeks)

**Goal:** Test on established benchmarks that require content-based retrieval and reasoning.

#### 2.1 — Task Selection

Tasks that specifically require Level 3+ capabilities, with long documents:

**Multi-hop QA (HotpotQA):** "What is the capital of the country where [person] was born?" Requires retrieving the person's birth country, then retrieving that country's capital. Two-hop retrieval.

**Coreference (WinoBias / Winogrande):** "The trophy didn't fit in the suitcase because it was too big." Requires matching "it" to "trophy" based on semantic reasoning about size.

**Long-range dependency (SCROLLS):** Tasks specifically designed to test long-context understanding — summarization, QA, NLI over multi-thousand-token documents.

**Selective copy / associative recall:** Synthetic tasks where specific tokens must be recalled based on content cues. Standard SSM stress tests.

#### 2.2 — Training Protocol

For each task:
- Train all-slow PRISM (no memory) as the lower bound
- Train HybridPRISM (with memory) as the test condition
- Train a parameter-matched Transformer as the upper bound (where feasible)
- Report accuracy, throughput, and memory usage

#### 2.3 — Memory Slot Analysis

After training, analyze what the memory slots learn:
- Visualize attention patterns: which tokens write to which slots? Which tokens read from which slots?
- Do slots specialize? (e.g., slot 1 captures entities, slot 2 captures relations, slot 3 captures temporal information)
- How does slot utilization vary across tasks?
- What happens when you reduce k from 64 to 16 to 4? At what point does performance degrade?

This analysis is important for understanding the model, and makes for compelling figures.

### Phase 3: Language Modeling (1-2 months)

**Goal:** The real test — can HybridPRISM match Transformer perplexity on autoregressive language modeling?

This is where most SSM architectures fall short. Mamba, RWKV, and others are competitive but consistently trail Transformers by a small margin on perplexity benchmarks. The memory bank could close this gap.

#### 3.1 — Causal Architecture Changes

For autoregressive LM, the architecture needs modifications:
- Recurrence becomes unidirectional (causal)
- Memory write must be causal — slot updates at position t can only use tokens 1..t
- Memory read must be causal — tokens can only query memory written by earlier positions
- This requires restructuring the memory as a streaming buffer, not a global bank

This is a significant implementation challenge. The bidirectional memory write used in Phase 1-2 won't work here.

**Causal memory design:**

```
For each chunk of w tokens:
  1. Run PRISM recurrence over chunk (O(w·d²))
  2. Write chunk summary to memory (O(w·k·d))
  3. Run memory read: tokens in chunk query memory from ALL previous chunks (O(w·k·d))
  4. Update memory with current chunk's information (O(k·d))
```

The memory bank grows in information content as the sequence progresses but maintains fixed size k. Older information gets compressed or overwritten based on the gated write mechanism. This is analogous to how human working memory operates — fixed capacity, content-addressable, with old items displaced by new ones.

#### 3.2 — Small-Scale LM Experiments

- Train on a standard corpus (OpenWebText, C4 subset, or similar)
- Model sizes: 25M, 80M, 150M parameters
- Compare HybridPRISM vs Transformer vs Mamba at matched parameter count and training compute
- Evaluate perplexity, downstream task accuracy, and throughput

#### 3.3 — Long-Context LM

The killer experiment:
- Train HybridPRISM with context length 8K, 16K, 32K
- Compare against Transformer with equivalent context (using FlashAttention, RoPE, etc.)
- Measure both quality and throughput
- The hypothesis: HybridPRISM matches quality at short contexts and wins at long contexts because the Transformer's quadratic cost forces smaller batches or shorter effective context

---

## Design Decisions and Open Questions

### How many memory slots (k)?

Start with k=32 and sweep {16, 32, 64, 128}. There's a tradeoff:
- Too few slots → information loss, can't represent complex documents
- Too many slots → approaches full attention cost, loses the sub-quadratic advantage
- The sweet spot is probably k = O(sqrt(n)) or k = O(log(n)), but this needs empirical validation

At k=32, n=8192: the memory read is 256x cheaper than full attention. At n=32K: 1024x cheaper. Even at k=128, the savings are 64x and 256x respectively.

### Where to place memory operations?

Current plan: after every 4 PRISM layers. Could also try:
- After every layer (maximum information flow, higher cost)
- Only once in the middle (minimal cost, might not be enough)
- Adaptive: a learned gate that decides whether to access memory at each layer

### How to initialize memory slots?

Three options:
- Learned parameters (current plan) — slots start as learned templates
- Zero initialization — slots start empty, filled entirely by writes
- Input-dependent initialization — first w tokens used to seed the memory

Learned initialization is the safest default. It gives each slot a "prior" about what to attend to.

### Single write/read or iterative?

The current design does one write then one read. An alternative:
- Write, read, write again, read again (iterative refinement)
- This lets the model correct its memory contents based on what the read step retrieved
- More expensive but potentially more powerful for multi-hop reasoning
- Test iterative refinement only if single-pass is insufficient

### Memory persistence across layers vs fresh at each layer?

Current design: memory persists across the two write/read operations (gated update at each write). Alternative: fresh memory at each memory layer, independent of the other. Persistent memory is more expressive but harder to train. Start with persistent, fall back to independent if training is unstable.

---

## Success Criteria

### Phase 1 (proof of concept)
- HybridPRISM matches all-slow PRISM on LoCoV1 embeddings (no regression)
- HybridPRISM solves needle-in-a-haystack at 8K+ where all-slow PRISM fails
- Latency overhead from memory is <10% vs pure recurrence

### Phase 2 (reasoning tasks)
- HybridPRISM beats all-slow PRISM on multi-hop QA by >10 points
- HybridPRISM is within 5 points of Transformer on reasoning tasks
- HybridPRISM maintains linear scaling advantage at 4K+ tokens

### Phase 3 (language modeling — the real test)
- HybridPRISM matches Transformer perplexity within 0.5% at matched compute
- HybridPRISM matches or beats Transformer at context lengths >8K
- HybridPRISM achieves >2x throughput advantage at 16K+ context

### Breakthrough criterion
HybridPRISM matches Transformer quality on standard LM benchmarks while scaling linearly. This would be the first sub-quadratic architecture to fully close the quality gap on general language tasks.

---

## Risk Factors

**Memory write learning dynamics.** The model must learn *what* to write to memory without direct supervision. The training signal is backpropagated through the memory read → task loss path, which is indirect. If the write mechanism doesn't learn useful compression, the memory bank will be ignored and HybridPRISM degrades to all-slow PRISM. Mitigation: auxiliary loss encouraging memory slot diversity.

**Causal memory for LM (Phase 3).** Converting the bidirectional memory mechanism to a causal streaming version is non-trivial. The gated write must make irreversible decisions about what to store with limited future context. This is the hardest engineering challenge and might require chunked processing.

**k selection.** Too few slots and you lose information. Too many and you approach quadratic cost. The right k may be task-dependent, which weakens the "general architecture" claim. Mitigation: show robustness across a range of k values.

**Comparison to existing work.** Memory-augmented networks (NTM, DNC, Memorizing Transformers) have been explored before. The novelty claim needs to be precise: it's the combination with linear-time recurrence, the specific write/read architecture, and the sub-quadratic total cost. Frame carefully.

---

## Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| Implementation | 1 week | Working HybridPRISM with memory bank |
| Phase 1: Proof of concept | 1-2 weeks | Needle-in-haystack results, LoCoV1 regression test |
| Phase 2: Reasoning tasks | 2-4 weeks | Multi-hop QA, coreference results |
| Phase 3: Language modeling | 1-2 months | Perplexity benchmarks, scaling curves |

Total: ~3 months to Phase 3 results.

---

## Relationship to Prior Work

**Neural Turing Machines / Differentiable Neural Computer (Graves et al.):** The original memory-augmented networks. Used content-based and location-based addressing with external memory. HybridPRISM's memory bank is conceptually similar but simpler — no read/write heads, no memory allocation. The cross-attention mechanism replaces the complex addressing scheme.

**Memorizing Transformers (Wu et al., 2022):** Augmented Transformers with a kNN memory bank of past hidden states. Similar motivation but different approach — they store raw states and retrieve via kNN, we compress into fixed-size slots via cross-attention writes.

**Mamba / Mamba-2:** State-of-the-art SSM for language modeling. Uses input-dependent gating (selectivity) but no explicit memory bank. HybridPRISM adds content-based retrieval on top of linear recurrence.

**Jamba (AI21):** Hybrid Mamba-Transformer interleaving Mamba layers with attention layers. Similar high-level structure to our proposal. Key difference: Jamba uses full quadratic attention at the attention layers. We use O(n·k) cross-attention to a memory bank, keeping the entire architecture sub-quadratic.

**RWKV:** Linear attention approximation using recurrence. No explicit memory mechanism. Competes with Transformers on perplexity but falls short on tasks requiring precise retrieval.

The positioning: HybridPRISM combines the efficiency of SSMs (linear-time recurrence for bulk processing) with the retrieval capability of Transformers (content-based addressing via memory) while maintaining sub-quadratic total cost. The memory bank is the bridge between these two paradigms.
