# Fine-Tuning LLMs for Chatbots with LoRA on Your Home Desktop


## Introduction

Large Language Models (LLMs) are the "rocket science" of our era. However, while a hobbyist could build a small rocket at home, pre-training even a modest LLM remained an unreachable far-flung goal for home desktops—until the game-changer Low-Rank Adaptation (LoRA) came into play.

LoRA fine-tunes GPT-2 within a reasonable time (a couple of hours) on your desktop. The process mirrors traditional LLM training but is far more affordable, giving you the rich flavor of the original LLM training experience.

This tutorial uses GPT-2 as our LLM and a 10,000 question-answer dataset. Basic GPT-2 (except for the specialized GPT2ForQuestionAnswering variant) cannot answer questions. We teach it within this tutorial, transforming it into a capable chatbot.

**Results:**
- Loss: 10.0 → 0.28 in 3 epochs
- Training time: 5 hours on CPU
- Adapter size: 2 MB
- 147K trainable parameters (99.8% reduction)

---

## What is LoRA?

LoRA (Low-Rank Adaptation) is an efficient machine learning technique used to fine-tune large, pre-trained AI models (like LLMs or Stable Diffusion) without modifying the entire original model. By freezing the original weights and adding small, trainable "low-rank" matrices to the network, LoRA significantly reduces training time, memory usage, and file sizes, making it possible to customize models on consumer hardware.

**Key Aspects of LoRA:**

- **Efficiency:** Instead of retraining billions of parameters, LoRA trains only a tiny fraction of new parameters
- **Small Files:** LoRA adapters are typically 2-300MB, compared to several GB for full models
- **Versatility:** Used to teach AI new styles, characters, or concepts without full retraining
- **Modular:** Multiple LoRAs can be applied to a base model and toggled or combined

---

## Why LoRA Works: Understanding LLM Architecture

Let's first understand how LLMs are structured. The diagram below shows a simplified architecture that illustrates the key principles. Each LLM consists of at least two main components: **a tokenizer** and **transformer blocks**.

### 1. Tokenization and Encoding Process

Text input is first processed by the tokenizer. Since neural networks work only with numbers, text must be converted into numerical sequences. The tokenizer handles this task.

**Building a vocabluary of tokens:**


# **Building a Vocabulary of Tokens**

Let's start with a simplified task: digitalize all English words by giving each word a unique ID. To build such a vocabulary of "word → ID" mappings, we download the entire English Wikipedia corpus and walk through it, assigning new IDs to unseen words. Once finished, we find that the vocabulary size (and the corresponding number of IDs) is about **half a million**! As we'll see later, this word-level approach results in huge memory consumption.

Now consider an alternative approach to word digitalization. The English language has approximately **10,000-15,000 base verbs** (though with all forms—present, past, continuous, compounds—this expands to 50,000-100,000+). Looking at regular verbs, we notice a pattern: instead of storing separate IDs for "play," "played," and "playing," we could store just the base form "play" plus two suffix tokens: "ed" and "ing." This way, any verb form requires only 2 token IDs: one for the base and one for the suffix.

**Example reduction:**
- Word-level: "play" (ID: 1), "played" (ID: 2), "playing" (ID: 3) → 3 IDs
- Token-level: "play" (ID: 1), "ed" (ID: 2), "ing" (ID: 3) → Can represent all forms with just these 3 tokens!

This splitting reduces vocabulary size. Continuing this process—finding and splitting words by the most frequent letter combinations and assigning IDs to them—we drastically reduce the space needed to represent all English words as combinations of token IDs in one vocabulary.

This procedure is called **tokenization**, where each derived letter combination is a **token** with its unique ID. Each model (GPT-2, LLaMA, BERT, etc.) has its own tokenization method, but the general rule for assigning token IDs is:

- **IDs 0-255:** Individual bytes/characters (all your keybord, to ensure any text can be represented)
- **IDs 256-1000:** Very common tokens ("the," "ing," "ed," etc.)
- **IDs 1000+:** Less common tokens

The total number of tokens is typically around **50,000**, which is **10 times less** than the number of words in the English language. Building the vocabulary is done once at the beginning, before or during the initial phase of model training. Once built, the vocabulary of tokens and token IDs remains the same for all model runs.

---

# **Tokenization of Input Text**

After building vocabluary, we can encode any text. Here is howmit works. After a text comes to the model's input, the following happens:

1. **Tokenization:** Text is split into tokens from the vocabulary
2. **ID mapping:** Tokens are replaced with their corresponding IDs
3. **Matrix creation:** A zero matrix is created where:
   - **Columns** = number of tokens in the text
   - **Rows** = total vocabulary size
4. **One-hot encoding:** For each column (token position), the element at row corresponding to the token's ID is set to 1; the rest remain zero
5. **Output:** Encoded matrix

---

## **Example:** Input text: `"Black cat sits on the mat"`

1. **Input:** "Black cat sits on the mat"
2. **Tokenize:** ["Black", " cat", " sit", "s", " on", " the", " mat"] (7 tokens)
3. **Assign IDs:** [9915, 3797, 1650, 82, 319, 257, 2603] *(These are token IDs according to GPT2 tokenizer)*
4. **Create matrix:** 7 columns × 50,000 rows (assuming 50,000 vocabulary size)
5. **One-hot encode:** 
   - Column 1, row 9915 = 1 (all other rows in column 1 = 0)
   - Column 2, row 3797 = 1 (all other rows in column 2 = 0)
   - And so on...

**Result:** A sparse matrix of shape `[50,000 × 7]` where only 7 elements are 1, and the rest are 0.

---

# **2. Embedding Layer**

As you can see, the encoded matrix size depends heavily on vocabulary size. Even with tokenization reducing the vocabulary from 500,000 words to 50,000 tokens, we still have a problem: a one-hot encoded matrix is extremely **sparse and memory-inefficient**.

**The problem:** 
- Original text: ~0.5 KB
- After one-hot encoding: 7 tokens × 50,000 vocab size × 4 bytes = ~1.4 MB for just 7 tokens!
- For a paragraph with 100 tokens: ~20 MB

To solve this, LLMs use an **embedding layer**—a learned lookup table that converts sparse one-hot vectors into dense, compact representations.

---

## **How Embedding Works:**

Instead of storing the full one-hot encoded matrix, we **multiply it by an embedding matrix**:

**Embedding matrix shape:** `[vocab_size × d_model]`
- **Rows:** Vocabulary size (e.g., 50,000)
- **Columns:** Embedding dimension `d_model` (typically 384-1024; we'll use 768 for GPT-2)

**Mathematical operation:**

Formally, each matrix defines a linear space and when we multiply matrix ***A*** by matrix ***B*** we formally map ***A*** in the space of ***B***. Thus we map encoded text from sparse space into compact and dence one, see figure 1. This mappin is called embedding.

![Sample Output](https://github.com/Vlasenko2006/LoRA_study/blob/main/figs/embeddings.png)
***Figure 1:*** Schematic view on embedding. Huge matrix of encoded text is multiplied by a embedding's layer matrix resulting in a small embedded text.


---

## **Memory Savings:**

**Before embedding (one-hot):**
- 7 tokens × 50,000 vocab size × 4 bytes = 1.4 MB

**After embedding:**
- 7 tokens × 768 dimensions × 4 bytes = 21.5 KB

**Over 60× reduction in memory!**

---

## **Which Other Benefits Brings Embedding Besides Memory Efficiency?:**

1. **Semantic meaning:** Similar words have similar embeddings (e.g., "cat" and "kitten" have similar vectors).
2. **Learnable:** The embedding matrix is trained with the model to capture meaningful relationships.
3. **Information preservation:** Despite dimensionality reduction, the embedding is learned to preserve relevant information for the task.

---
## **Positional Encoding**

When we read text, we naturally process it from beginning to end, building a logical sequence of events in our minds as we follow the word order. However, **transformers process all tokens simultaneously in parallel**—they see the entire input at once, which means they have **no inherent sense of word order or position**.

**The problem:** Without positional information, the transformer would treat these sentences identically:
- "The cat chased the dog" 
- "The dog chased the cat"

Both have the same tokens, just in different positions—but the meaning is completely different!

**The solution:** To preserve word order information, we add **positional encoding** to the embedded tokens. This encoding injects information about each token's position in the sequence directly into its representation, allowing the transformer to "see" the text flow.

Positional encoding creates a matrix `PE` with shape `[sequence_length × d_model]` where each row contains a unique pattern that represents a specific position in the sequence. This matrix is then **added** (element-wise) to the embedded token matrix:


```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    # even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    # odd dimensions
```

where `pos` is the position of the token in the text (0, 1, 2, ..., seq_len-1) and `i` ranges over the embedding dimensions (i = 0, 1, 2, ..., d_model-1), with even indices using sine and odd indices using cosine. To see how it works cosnider:

**Positional embedding of a string (For simplicity, assume that each word is a token):** "What is your name? My name is Alex". 


```
Tokens: ["what", "is", "your", "name", "?", "My", "name", "is", "Alex"]
Positions: [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

Figure 2 visualizes the positional encoding patterns for this sentence. Panel (a): Full positional encoding matrix showing all tokens and embedding dimensions. Each row represents a unique wave pattern for a specific position. Panel (b): First 64 dimensions showing the sine/cosine wave patterns more clearly. Notice how each position has a distinct pattern. Panel (c): Position similarity matrix showing how positional encodings relate to each other. The diagonal (brightest) shows that each position is most similar to itself. Off-diagonal elements show that nearby positions have higher similarity than distant positions. Panel (d): Cross-sections of wave patterns for selected tokens (positions 0, 2, 4, 6, 8). Each curve shows how the encoding values vary across dimensions for that specific position.

**Key observations:**

- **Each position gets a unique wave signature:**
  - Position 0 ("what") has a distinct pattern
  - Position 4 ("?") has a different pattern
  - Position 8 ("Alex") has yet another pattern

- **Same token, different positions have different patterns:**
  - "is" at position 1 (after "what") has a different wave pattern than "is" at position 7 (after "name")
  - "name" at position 3 has a different pattern than "name" at position 6
  - This allows the model to distinguish between identical tokens based on their position

- **Wave patterns vary across dimensions:**
  - Early dimensions (left side of panels a, b) oscillate quickly
  - Later dimensions (right side) oscillate slowly
  - This multi-frequency approach encodes position information at different scales.

By adding these positional encodings to the token embeddings, the transformer can now distinguish "is" at position 1 from "is" at position 7, even though the tokens themselves are identical. The model learns to use this positional information to understand word order and sequence structure.


![Sample Output](https://github.com/Vlasenko2006/LoRA_study/blob/main/figs/Attention_is_all_you_need_768.png)
***Figure 2:*** Scheme of positional embedding of a sentence "What is your name? My name is Alex". For simplicity each word and punctiation is a token.  Each position gets its unique wave pattern. Compare two "is" wave patterns after "What"  and "name" on subfigures A and B. Compare also in these subbfigures wave patterns for two "name". Subfigure D shows the cross-section of wave patterns of some tokens. Subfigure C shows the Position similarity matrix. Dot product of tokens postions of keys and queries. It shows how far each tokes stays from the others.  

---

#### **3. Transformer Architecture**

The core of an LLM consists of stacked transformer layers. The number of layers determines model quality, context understanding, and response quality:
- **Simple models:** 6 layers
- **Advanced models:** 12, 24, 48+ layers

Each transformer has three key sublayers:

##### **a) Attention Sublayer**

The attention mechanism identifies context, main points, and relationships between tokens. It processes the embedded text matrix by multiplying it with **attention heads**—specialized matrices that learn specific text patterns during training.

**How attention heads work:**

Each attention head consists of multiple matrices `Q,K,V` (typically Query, Key, and Value matrices). When combined, they:
- Identify relationships between tokens
- Weight token importance based on context
- Capture semantic meaning and dependencies

**Example for `Q,K,V`:** Let `E` be the compressed matrix. A question-detection head might (but not must) work as follows:
We compute query and key matrices as follows `Q = E · W_q`, `K = E · W_k`, where `W_q, W_k` are trainable query and key matrices respectively.

- Key matrix `K` assigns high weights to embeddings corresponding to question indicators, i.e., words like "what", "where", "which", question marks and auxiliary verbs, like "does/do", "is/are". 
- Query matrix `Q` assigns high values to verb and subject tokens, since their presence and position strictly affect the type of sentence, i.e., verb conjugation and subject/verb word order changes in assertions and questions.
- Multiplication `QK^T` does the first magic. Question indicators meet verb and subject embeddings giving multiplicative high scores. And here our `PE` plays a crucial role. Without `PE`, question words like "What" would give the same high score for any auxiliary verb "is", wherever it appears in the text.  But `PE`, introduces the scores of order. Since `PE` is a part of `E`, expanding `QK^T` we get that it its proportional to `PE*PEˆT`. Panel C in figure 2 shows `PE*PEˆT` which literally scores how far each word (off main diagonal ,score <1) in the text stays away from the current word (main diagonal, score 1). The Transformer learns how `PE`, and `PE*PEˆT` works and becomes completely aware of token relations.
- The next step computes activation function `softmax(QK^T / sqrt(d_k))`, where `sqrt(d_k)` is the normalization factor. 
- Here occurs the final magic where the activation function output is multiplied by matrix of values `V`. The attention weights score how much each position should attend to others. If "What" (question indicator) and "is" (verb) have high attention score AND are at specific relative positions, "is" receives strong signal from V["What"], inheriting the "this is a question" context.


![Sample Output](https://github.com/Vlasenko2006/LoRA_study/blob/main/figs/with_without_PE.png)
***Figure 3:*** `QK^T` Matrices with and without positional embedding. The embedded text is "What is your name? My name is Alex". For simplicity each word is a token, `Q` contains ones for question markers "What", "name","?"; `K` contains one for auxiliary verb "is"; all other elements in both matrices are zeros. Positional embedding shows clear relations between different words.



With this question attention head, the model understands whether a question was asked and what was asked. The multi-head attention mechanism uses multiple heads simultaneously to capture different aspects of meaning (e.g., temporal context, spatial context, causality).

**Output:** Contextualized vectors representing the meaning of each token in relation to others.

#### **b) Feed-Forward Sublayer**

This is the "thinking" layer that analyzes contextualized vectors and makes decisions. It consists of:
- Linear or non-linear activation functions (typically ReLU or GELU)
- Formula: `f(a₁x₁ + a₂x₂ + ... + aₙxₙ)` where:
  - `a₁, ..., aₙ` are trainable weights
  - `x₁, ..., xₙ` are elements from the attention output

**This is where LoRA focuses its fine-tuning**, as this layer contains the model's decision-making logic.

#### **c) Normalization Sublayer**

Normalizes the feed-forward output using a specific rule (e.g., layer normalization, spectral normalization). This prevents gradients from exploding or vanishing during training.

---

### 4. How LoRA Fine-Tunes Transformers

Now that we understand transformer basics, let's see where LoRA fits in:

**Key insight:** 
- The **attention layer** is already properly trained for understanding context—fine-tuning it makes little sense
- The **feed-forward layer** makes decisions about context—this is what we should fine-tune

The feed-forward matrix can be thought of as having "directions of thinking" (mathematically, these are eigenvectors). Since fine-tuning datasets are much smaller than pre-training datasets, we only need to modify a few of these directions.

**LoRA's approach:**

1. **Freezes ALL pre-trained model weights** (no modification to original parameters)
2. **Injects trainable low-rank matrices** into transformer layers between feed-forward and normalization sublayers
3. **Reduces trainable parameters by 99%+** while maintaining performance

---

### The Math Behind LoRA

**Original weight update:**
```
W_new = W_frozen + ΔW
```

**LoRA approximation:**
```
ΔW ≈ (lora_alpha/r) × B × A

where:
  B: (d × r) trainable matrix
  A: (r × k) trainable matrix
  r: rank (typically 4-16)
  lora_alpha: scaling factor
```

**Key principle:** Fine-tuning updates exist in a low-dimensional subspace, so we don't need full-rank updates. LoRA exploits this by decomposing the weight update into two small matrices (B and A), drastically reducing the number of trainable parameters.

---

## Next Steps

Continue to the [LoRA Fine-Tuning Tutorial](LoRA_Fine_Tuning_Tutorial.ipynb) for hands-on implementation with code examples and practical exercises.
