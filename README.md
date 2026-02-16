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

## **Example:** Input text: "Black cat sits on the mat"

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
E = E_0 + PE 
```

where `pos` is the position of the token in the text (0, 1, 2, ..., seq_len-1) and `i` ranges over the embedding dimensions (i = 0, 1, 2, ..., d_model-1), with even indices using sine and odd indices using cosine and `E_0` is the original embedded text matrix. 

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

## **Example: How Query, Key, and Value Matrices Work**

Let `E` be the embedded matrix (token embeddings + positional encodings). A question-detection attention head might work as follows:

We compute query and key matrices: 

```
Q = E · W_Q
K = E · W_K
```

where `W_Q` and `W_K` are trainable weight matrices.

### **Role of Each Matrix:**

- **Key matrix `K`:** Identifies question indicators by assigning high weights to tokens like "what", "where", "which" and question mark "?".

- **Query matrix `Q`:** Identifies sentence structure elements by assigning high values to verbs and subjects, since their presence and position determine sentence type (e.g., verb conjugation and word order differ between statements and questions).

- **Value matrix `V`:** Encodes what information to extract. For example, it might assign high weights to question marks and varying weights (0.3, 0.6, 0.9) to different question types: binary (yes/no), multiple-choice, or open-ended questions.

### **The First Magic: QK^T Multiplication**

The multiplication `QK^T` computes combination scores between key and query tokens. If question indicators (from `K`) meet structural elements (from `Q`) at right place (i.e, if "When" token from `K` is immediately follwed with query token "is"), they produce high scores. 

**Here, positional encoding plays a crucial role.** Let's see why with an example.

### **Example: "What is your name? My name is Alex"**

For simplicity, assume each word and punctuation mark is a token:

```
Tokens: ["what", "is", "your", "name", "?", "My", "name", "is", "Alex"]
```

**Simplified setup:**
- **Query `Q`:** Only question markers have values: "what" (pos 0), "?" (pos 4)
- **Key `K`:** The auxiliary verb "is" and subject "name" have values (pos 1, 3, 6, 7)
- Assume for simplicity that all  these values equal to one
- All other elements are zero

**Figure 3** shows two scenarios:

#### **Panel A: Without Positional Encoding (QK^T where E = E_0 )**

When we use only token embeddings (no positional information), all combinations of {"what", "?"} × {"is", "name"} get similar scores:
- "what" × "is" (pos 1) ≈ "what" × "is" (pos 7) — **Same score!**
- "name" (pos 3) × "?" (pos 4) ≈ "?" (pos 4) × "name" (pos 6) — **Same score!**

**Problem:** The model cannot distinguish which "is" belongs to the question and which belongs to the answer. There's no understanding of word relationships or sentence boundaries. Similar situation with "name" token.

#### **Panel B: With Positional Encoding (QK^T where E = E_0 + PE)**

Now, pairs belonging to the same sentence (the question) get **significantly higher scores**:
- "what" × "is" (pos 1): **High score** ✓ (same sentence, distance = 1)
- "what" × "is" (pos 7): **Low score** (different sentence, distance = 7)
- "name" (pos 3) × "?" (pos 4): **High score** ✓ (question sentence, distance = 1)
- "name" (pos 6) × "?" (pos 4): **Lower score** (answer sentence, distance = 2)

**Why does this happen?**

Substituting `E = E_0 + PE` into the expressions for `Q` and `K`:

```
Q = (E_0 + PE) · W_Q = E_0 · W_Q + PE · W_Q
K = (E_0 + PE) · W_K = E_0 · W_K + PE · W_K
```

Expanding `QK^T`:

```
QK^T = (E_0 · W_Q + PE · W_Q)(E_0 · W_K + PE · W_K)^T
     = E_0 W_Q W_K^T E_0^T + E_0 W_Q W_K^T PE^T + PE W_Q W_K^T E_0^T + PE W_Q W_K^T PE^T
```

The **positional component** `PE · W_Q · W_K^T · PE^T` is proportional to `PE · PE^T`!

**Look at Panel C in Figure 2** showing `PE · PE^T`: This matrix encodes **positional similarity**—how close tokens are to each other in the sequence:
- **Diagonal:** Maximum values (each token compared to itself)
- **Near-diagonal:** High values (nearby tokens)
- **Far from diagonal:** Lower values (distant tokens)

Thus, `QK^T` combines:
1. **Semantic similarity** (from token embeddings)
2. **Positional proximity** (from positional encodings)

This gives **highest scores to semantically related tokens that are also nearby**, allowing the model to distinguish the question from the answer!

### **The Text Flow Direction Problem**

Although the matrix `PE · PE^T` encodes token order and relative distances in the sequence, it has a critical limitation: **it is symmetrical**. This symmetry means the transformer cannot inherently distinguish the direction of text flow—where it starts versus where it ends.

**Understanding the symmetry problem:**

Think of the `PE · PE^T` matrix in human terms:
- **The diagonal** represents the "present moment" for each token
- **Below the diagonal** (row > column) represents tokens looking backward at earlier context
- **Above the diagonal** (row < column) represents tokens looking forward at future context

However, because `PE · PE^T` is symmetrical, the relationship between position (i, j) equals the relationship between position (j, i). The matrix treats "token 5 attending to token 2" identically to "token 2 attending to token 5"—it only knows they are 3 positions apart, not which comes first.

**The solution: Breaking symmetry with causal masking**

To teach the transformer text directionality, we add a **mask matrix** `M` to `QK^T` that breaks the symmetry:

```
Attention_scores = QK^T / √d_k + M
```

The causal mask `M` is an upper-triangular matrix:

**Effect of the mask:**
- Values of `-∞` force attention weights to zero for future positions (after softmax)
- Each token can only attend to itself and **previous** tokens (below and on the diagonal)
- This enforces left-to-right information flow, making text direction explicit

**Alternative approach: ALiBi (Attention with Linear Biases)**

Instead of masking, some modern architectures (like those using ALiBi) add a **slope matrix** that explicitly encodes distance with direction. 
Each attention head learns a slope and the corresponding bias penalizes distant tokens while preserving directionality. This approach eliminates the need for positional embeddings entirely while making text flow direction mathematically explicit.



### **The Second Step: Softmax Normalization**

We apply the softmax function:

```
Attention_weights = softmax(QK^T / √d_k + M)
```

where `√d_k` is a scaling factor (square root of the key dimension) that prevents extremely large values from causing numerical instability in the softmax function. Softmax converts raw scores into a probability distribution, ensuring all attention weights sum to 1 for each query token.

### **The Final Magic: Multiplying by Value Matrix V**

The attention weights determine **how much each token should attend to other tokens**. We compute:

```
Output = Attention_weights · V
```

**How `V` works:** Each column of `V` represents the input sequence with weighted token representations. For example, in a question-detection head:
- Question marks "?" get weight 1.0
- Binary question words ("is", "are", "do") get weight 0.3
- Multiple-choice indicators get weight 0.6
- Open question words ("what", "where", "why") get weight 0.9

**Example computation:**
- The token pair "What is" has a nearby question mark "?" → gets high attention weight
- This high attention weight multiplied by `V` (which assigns 0.9 to "what"-type questions) → produces a strong signal
- The model learns: "This is an open-ended question requiring a detailed answer"

Thus, the transformer not only recognizes **"this is a question"** but also learns:
1. **Question type** (binary, multiple-choice, or open-ended)
2. **Question importance** to the overall context
3. **What kind of answer is expected**

This attention mechanism allows the model to build rich, context-aware representations that capture both syntax and semantics!
