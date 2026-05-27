# Tokens in Large Language Models: Complete Guide

> **One Sentence**: A token is the basic unit of text that an LLM processes—not necessarily a whole word, but a "chunk" of characters that the model converts to numbers before understanding.

---

## Table of Contents
1. [What is a Token? (The Core Concept)](#what-is-a-token-the-core-concept)
2. [How Tokenization Works Internally](#how-tokenization-works-internally)
3. [From Text to Numbers: Tokenization & Embeddings](#from-text-to-numbers-tokenization--embeddings)
4. [Token Economics: Costs & Context Windows](#token-economics-costs--context-windows)
5. [Hidden Token "Taxes" (What Wastes Your Budget)](#hidden-token-taxes-what-wastes-your-budget)
6. [Language Impact: English vs Non-English](#language-impact-english-vs-non-english)
7. [Token Counting & Estimation](#token-counting--estimation)
8. [Advanced: How Tokenizers Are Built](#advanced-how-tokenizers-are-built)
9. [Embeddings Deep Dive](#embeddings-deep-dive)
10. [Optimization Strategies](#optimization-strategies)
11. [Real-World Examples](#real-world-examples)
12. [Interview Preparation](#interview-preparation)

---

## What is a Token? (The Core Concept)

### Simple Definition
A token is the basic unit of text that an LLM reads and understands. It's **not necessarily a whole word**. Instead, think of it as a "chunk" of characters that the model treats as a single atomic unit.

### How It Works
The model breaks down your prompt into these small pieces **before processing it**.

### Token Examples

| Text | Tokens | Count | Notes |
|------|--------|-------|-------|
| "hello" | hello | 1 | Common word = 1 token |
| "learning" | learn, ing | 2 | Uncommon word = split into parts |
| "hello world" | hello, world | 2 | Space is included in token boundaries |
| "I'm" | I, ', m | 3 | Apostrophe creates extra token |
| "🚀" | 🚀 | 1 | Emoji = 1 token |
| "123" | 123 | 1 | Numbers can be single tokens |
| "code123" | code, 123 | 2 | Mixed = multiple tokens |

### Why Not Just Process Letters?

❌ **Why character-by-character doesn't work:**
- Processing each letter individually would be SLOW
- A 1000-word document = 5000+ characters
- The model would need to learn patterns from scratch for every 5000-character sequence
- Massive computational overhead

✅ **Why tokens are the sweet spot:**
- Average word = 1.3 tokens
- Balances **speed** (faster than characters) and **understanding** (meaningful chunks)
- The model can learn patterns from ~50k tokens instead of ~100k characters
- Efficiency multiplier: ~10-50x faster processing

### Live Demo: OpenAI Tokenizer Tool

You can paste text into the OpenAI Tokenizer (https://platform.openai.com/tokenizer) and see exactly how many tokens it becomes.

**Example:**
```
Input: "Hello, how are you doing today?"
Output: 8 tokens
Breakdown: [Hello] [,] [how] [are] [you] [doing] [today] [?]
```

**Longer Example:**
```
Input: A short paragraph of 100 words
Output: 120-150 tokens (words ≠ tokens)
```

---

## How Tokenization Works Internally

### The Tokenization Pipeline

```
Raw Text Input
    ↓
Step 1: Preprocessing
├─ Normalize whitespace
├─ Handle special characters
└─ Prepare for encoding
    ↓
Step 2: Token Selection (Vocabulary Lookup)
├─ Match longest known token
├─ Fall back to smaller chunks
└─ Handle unknown characters as special tokens
    ↓
Step 3: Token ID Assignment
├─ Convert token to numeric ID
└─ Example: "hello" → [15145]
    ↓
Tokenized Output: [15145, 11, 921, 527, 291]
```

### Example: Tokenizing "Learning is fun!"

```
Step 1: Input Text
"Learning is fun!"

Step 2: Tokenizer breaks it down
├─ "Learning" → Check vocabulary
│  ├─ "Learning" not in vocab? → Check "Learn" → ✓ Found
│  └─ Remaining "ing" → Check vocab → ✓ Found
├─ "is" → Check vocabulary → ✓ Found (common word)
├─ "fun" → Check vocabulary → ✓ Found
└─ "!" → Check vocabulary → ✓ Found

Step 3: Result
Tokens: ["Learn", "ing", "is", "fun", "!"]

Step 4: Convert to IDs (using vocabulary)
Token IDs: [12379, 2233, 318, 1257, 0]

Step 5: Model processes these IDs
(Each ID is converted to a vector internally)
```

### The Tokenizer's Vocabulary

A typical LLM tokenizer has a **vocabulary size of 50,000-100,000 tokens**.

```
Token ID | Token | Frequency | Example Usage
---------|-------|-----------|---------------
1        | [UNK] | N/A       | Unknown token
2        | [CLS] | N/A       | Start of sequence
3        | " "   | Very High | Every sentence
100      | "the" | Very High | Common English word
5000     | "cat" | Medium    | Common noun
50000    | "🚀"  | Low       | Emoji
```

**Important**: The vocabulary is **fixed at training time**. The tokenizer learns which token chunks are most efficient based on the training data.

---

## From Text to Numbers: Tokenization & Embeddings

### The 2-Step Process

This is how text becomes mathematics that the model understands.

| Step | Process | What Happens | Output |
|------|---------|--------------|--------|
| **1: Tokenization** | Break text into chunks | Raw text → List of token IDs | [15145, 11, 921] |
| **2: Embedding** | Convert token ID to vector | Token ID → Numeric vector | [[0.1, -0.2, 0.5, ...]] |

### Visual Representation

```
Input Text: "The cat sat"
    ↓
TOKENIZATION (Text → Token IDs)
├─ "The" → Token ID: 262
├─ "cat" → Token ID: 1200
└─ "sat" → Token ID: 3891
    ↓
Result: [262, 1200, 3891]
    ↓
EMBEDDING (Token IDs → Vectors)
├─ Token 262 → [0.1, -0.5, 0.3, ..., 0.2]  (768 dimensions)
├─ Token 1200 → [0.4, 0.1, -0.2, ..., 0.9]  (768 dimensions)
└─ Token 3891 → [0.2, -0.1, 0.6, ..., 0.3]  (768 dimensions)
    ↓
Result: A 3×768 matrix that the neural network processes
```

### Why Not Just Use ASCII Codes?

❌ **ASCII approach (DOESN'T WORK for LLMs):**
```
"bank" (river) → ASCII: [98, 97, 110, 107]
"bank" (money) → ASCII: [98, 97, 110, 107]

Problem: Same ASCII codes! Model can't distinguish meaning.
```

✅ **Embedding approach (WORKS):**
```
"bank" (river) → Vector: [0.1, -0.5, 0.3, 0.8, ...]
"bank" (money) → Vector: [0.1, -0.5, 0.2, -0.4, ...]

The vectors are DIFFERENT!
They capture semantic meaning and context.
```

### Semantic Space: GPS Coordinates for Words

The embedding process places each word in a **multi-dimensional space** where similar concepts are near each other.

```
Conceptual Space (simplified to 2D for visualization):

                    ↑ POSITIVE
                    |
        "happy"     |    "joyful"
            ●───────┼────● 
                    |    
                    | 
    ────────────────●─────────→ POSITIVE
     "sad"      |      "angry"
        ●───────┼────●
                |
                ↓ NEGATIVE

In reality: 768 to 12,288 dimensions (depending on model)
```

**Key Property**: Words with similar meanings are **close together** in this space.

```
Vector distances:
distance("king", "queen") = 0.05 (very close - both royalty)
distance("king", "cat") = 0.8 (far apart - different concepts)
distance("king", "prince") = 0.15 (closer - related royalty)
```

### How Embeddings Capture Meaning

During training, the model learns:
- **Semantic meaning**: "dog" ↔ "animal" (similar vectors)
- **Context relationships**: "king" - "man" + "woman" ≈ "queen"
- **Sentiment**: "happy" is opposite of "sad"
- **Domain knowledge**: "Python" ↔ "programming"

All encoded in the **relative positions** of vectors in embedding space.

---

## Token Economics: Costs & Context Windows

### Context Window: The Model's Short-Term Memory

Every LLM has a **context window** limit (measured in tokens).

```
Context Window = Maximum tokens in a single request
                = Input + Output combined
```

| Model | Context Window | Real-World Equivalent |
|-------|---------------|-----------------------|
| GPT-3.5 | 4,096 tokens | ~3,000 words / ~2 pages |
| GPT-4 | 8,192 tokens | ~6,000 words / ~4 pages |
| GPT-4 Turbo | 128,000 tokens | ~96,000 words / ~65 pages |
| Claude 3 Opus | 200,000 tokens | ~150,000 words / ~100 pages |
| Gemini Ultra | 1,000,000 tokens | ~750,000 words / ~500 pages |

### What Happens When You Fill the Context?

```
Context Window: 128,000 tokens (GPT-4 Turbo)

Tokens Used:
├─ System prompt: 2,000 tokens
├─ Your question: 500 tokens
├─ Retrieved documents: 50,000 tokens
├─ Chat history: 30,000 tokens
├─ Model thinking space: 20,000 tokens
└─ Available for response: 25,500 tokens
                           ↑
                    Limited output!

If you add more context → Context fills up → Oldest parts get dropped
```

### Token Costs Are Asymmetric

Different types of tokens cost different amounts.

#### Input Tokens (Cheaper)
- Your prompt
- System instructions
- Retrieved data from tools/APIs
- Chat history
- **Cost multiplier: 1x**

#### Output Tokens (More Expensive)
- Model's generated response
- Takes more computation to produce
- **Cost multiplier: 3-5x more expensive** than input

### Real Example: API Costs

```
OpenAI GPT-4 Turbo Pricing (2026):
Input:  $0.01 per 1K tokens
Output: $0.03 per 1K tokens

Request:
├─ Input: 1,000 tokens → Cost: $0.01
└─ Output: 500 tokens → Cost: $0.015
   Total: $0.025 per request

Input-heavy task (like RAG): Better cost efficiency
Output-heavy task (like summarization): More expensive
```

### TPS: Tokens Per Second (Speed Metric)

How fast does the model generate tokens?

```
TPS = Tokens Per Second

Fast models: 100+ TPS
├─ GPT-4: ~50-80 TPS
├─ Claude: ~60-100 TPS
└─ Gemini: ~80-120 TPS

Slow models: <50 TPS
├─ o1 (reasoning): ~10-30 TPS (spends time thinking)
└─ Custom fine-tuned models: varies widely

Impact on user experience:
512 tokens at 100 TPS = 5 seconds to complete
512 tokens at 10 TPS = 51 seconds to complete
```

---

## Hidden Token "Taxes" (What Wastes Your Budget)

These are places where tokens are **silently consumed** without producing visible value.

### 1. System Prompts (The Biggest Waste)

**The Problem:**
- Your system prompt is sent with **EVERY API call**
- If it's 500 tokens, you waste 500 tokens per request

**Example:**
```
System Prompt (500 tokens):
"You are an expert AI assistant specializing in Python coding...
You should provide detailed explanations... Always format code 
as markdown... Never provide code that could be used for hacking..."

Usage pattern:
├─ Call 1: 500 tokens (system) + 100 tokens (user) = 600
├─ Call 2: 500 tokens (system) + 120 tokens (user) = 620
├─ Call 3: 500 tokens (system) + 150 tokens (user) = 650
└─ ... × 1000 calls = 500,000 wasted tokens on system prompt!

If output is 200 tokens per call:
Cost: 500,000 input tokens = $5.00 at OpenAI rates
```

**Solution:**
- Keep system prompts under 100 tokens
- Move instructions to user prompt when possible
- Use prompt caching (sends system prompt once per hour)

### 2. Chat History (The Conversation Bloat)

**The Problem:**
- The ENTIRE chat history must be re-sent with every message
- In long conversations, history can be 80% of your context

**Example:**
```
A 5-message conversation:
Message 1: 100 tokens (user) + 200 tokens (assistant)
Message 2: 80 tokens (user) + 150 tokens (assistant)
Message 3: 120 tokens (user) + 180 tokens (assistant)
Message 4: 90 tokens (user) + 160 tokens (assistant)
Message 5: 110 tokens (user) → Send for response

When sending Message 5:
Tokens sent: (100+200) + (80+150) + (120+180) + (90+160) + 110
           = 1,190 tokens JUST for history
           = 91% of budget for a 1,300 token request

Message 6: 95 tokens (user) → Send for response
Tokens sent: 1,190 (history) + 95 (new question) = 1,285
           = Same problem repeats!
```

**Solution:**
- Delete old messages or keep only last 3-5
- Summarize long conversations (another model call, but cheaper)

### 3. Tool / MCP Server Outputs (Data Bloat)

**The Problem:**
- Results from APIs are converted to tokens
- Large JSON responses = many tokens

**Example:**
```
API Call: get_weather("Paris")
Returns:
{
  "city": "Paris",
  "temperature": 22,
  "condition": "sunny",
  "humidity": 65,
  "wind": 5,
  "forecast": [...detailed 5-day data...],
  "alerts": [...],
  "historical": {...}
}

Original JSON: 2,000 characters
Tokens used: ~500 tokens

User only needed: "Paris is 22°C and sunny"
Could be: 1 token!

Waste: 499 tokens (~$0.005 per API call)
```

**Solution:**
- Filter API responses to only needed fields
- Post-process results before sending to LLM
- Use structured extraction instead of raw JSON

### 4. Structured Outputs (JSON Formatting Tax)

**The Problem:**
- JSON formatting adds many tokens (brackets, quotes, newlines)
- Natural language is more token-efficient

**Example:**
```
Same data, different formats:

Natural Language (Most efficient):
"Paris is 22°C, sunny with 65% humidity and 5 km/h winds."
Tokens: 28

JSON (Less efficient):
{
  "city": "Paris",
  "temperature": 22,
  "condition": "sunny",
  "humidity": 65,
  "wind": 5
}
Tokens: 32 (14% more!)

XML (Even less efficient):
<weather>
  <city>Paris</city>
  <temperature>22</temperature>
  ...
</weather>
Tokens: 45 (60% more!)
```

**When to use JSON anyway:**
- If you need to parse the output programmatically
- If you need guaranteed structure
- If the overhead is worth the certainty

### 5. Latent / Reasoning Tokens (The Hidden Thinking Cost)

**The Problem:**
- Some models (like OpenAI's o1) do internal reasoning
- You PAY for these tokens even though you don't see them

**Example:**
```
User Question: "Prove that the Riemann Hypothesis is correct"

Model Process:
├─ Internal reasoning tokens: 5,000 (thinking, exploring)
├─ Draft answer tokens: 1,000 (generating response)
└─ Final answer visible: 500 tokens

You see: 500 tokens of response
You pay for: 5,000 + 1,000 + 500 = 6,500 tokens!

Cost ratio: 13x what you see!
```

**Strategy:**
- Use reasoning models only for complex problems
- For simple questions, use faster models
- Factor in hidden reasoning costs to ROI calculations

### Summary: Token Tax Impact

```
Typical Request Breakdown (assuming 128k context):

Useful Tokens:        Input: 500 tokens (what user wanted to ask)
                      Output: 200 tokens (the answer)
                      ───────────────
                      Subtotal: 700 tokens (actual value)

Wasted Tokens:        System prompt: 500 tokens
                      Chat history: 2,000 tokens
                      Tool output: 1,500 tokens
                      Formatting overhead: 300 tokens
                      Reasoning (hidden): 2,000 tokens
                      ───────────────
                      Subtotal: 6,300 tokens (wasted!)

Total Tokens Used:    7,000 tokens
Efficiency:           10% useful, 90% waste!
```

---

## Language Impact: English vs Non-English

### The Critical Discovery: Language Matters

**The Word "hello"**

| Language | Text | Tokenization | Token Count | Cost Multiplier |
|----------|------|--------------|-------------|-----------------|
| English | hello | [hello] | 1 token | 1x |
| Spanish | hola | [hola] | 1 token | 1x |
| French | bonjour | [bon, jour] | 2 tokens | 2x |
| German | guten Morgen | [gut, en, Mor, gen] | 4 tokens | 4x |
| Russian | привет | [при, ве, т] | 3 tokens | 3x |
| Arabic | مرحبا | [م, ر, ح, ب, ا] | 5 tokens | 5x |
| Telugu | నమస్కారం | [న, మ, స, ్, క, ార, ం] | **9 tokens** | **9x** |
| Chinese | 你好 | [你, 好] | 2 tokens | 2x |

### Why This Happens

**Tokenizers are language-biased:**

1. **Training data bias**: Trained mostly on English text (80%+ of internet text)
2. **Vocabulary coverage**: 
   - English: 50,000 tokens cover ~99% of common usage
   - Telugu: 50,000 tokens cover ~60% of common usage
   - Result: Telugu uses multiple smaller tokens per word

3. **Unicode complexity**:
   - English: Simple ASCII (a-z = easily tokenizable)
   - Telugu: Complex Unicode combining characters
   - Example: "ా" (two Unicode code points) = might be 2-3 tokens

### Real Impact: Cost & Performance

**Scenario: A Telugu speaker using ChatGPT**

```
Telugu Sentence: "నా పేరు రాज"
English Equivalent: "My name is Raj"

English version:
├─ Tokens: 5 tokens
├─ Cost: $0.0005
└─ Generation time: 0.05 seconds

Telugu version:
├─ Tokens: 45 tokens (9x more!)
├─ Cost: $0.0045 (9x more!)
└─ Generation time: 0.45 seconds (9x slower!)

Annual cost difference (1 million conversations):
├─ English: $500
└─ Telugu: $4,500 (difference: $4,000)

The Telugu speaker pays 9x more for the same service!
```

### Which Languages Are Most Affected?

**Least Affected (Best tokenization):**
- English: 1x multiplier
- Spanish, French, German: 1-2x
- Russian (Cyrillic): 2-3x

**Moderately Affected:**
- Arabic: 3-5x
- Chinese, Japanese: 1.5-2x

**Most Affected (Worst tokenization):**
- Telugu, Tamil, Kannada: 8-12x
- Hindi, Marathi: 6-10x
- Vietnamese, Thai: 5-8x
- Rare scripts: Can be 20-30x!

### Why This Matters for Users

| Impact | Effect |
|--------|--------|
| **Cost** | Non-English speakers pay more for same service |
| **Context** | Fill context window 9x faster |
| **Speed** | Responses take 9x longer to generate |
| **Quality** | Model might perform worse (less training data coverage) |
| **Fairness** | Creates digital divide for non-English speakers |

### Solutions Being Developed

1. **Multilingual tokenizers**: Train on more diverse languages
2. **Byte-pair encoding improvements**: Better handling of rare scripts
3. **Language-specific models**: Optimize for specific languages
4. **Custom vocabularies**: Add language-specific tokens during fine-tuning

---

## Token Counting & Estimation

### How to Count Tokens Accurately

#### Method 1: OpenAI Tokenizer (Online Tool)

Visit: https://platform.openai.com/tokenizer

Paste your text and get instant token count.

**Limitations:**
- Only works for OpenAI models
- Doesn't account for system prompts or special tokens
- Manual process

#### Method 2: Programmatic Counting

```python
# Using OpenAI's Python library
from tiktoken import encoding_for_model

# Get tokenizer for specific model
enc = encoding_for_model("gpt-4")

# Count tokens in text
text = "Hello, how are you?"
tokens = enc.encode(text)
print(f"Token count: {len(tokens)}")  # Output: 8

# Get token details
for i, token_id in enumerate(tokens):
    token = enc.decode([token_id])
    print(f"{i}: {token_id} → '{token}'")
```

**Output:**
```
0: 15145 → 'Hello'
1: 11 → ','
2: 543 → ' how'
3: 527 → ' are'
4: 291 → ' you'
5: 30 → '?'
```

#### Method 3: Quick Estimation (Rule of Thumb)

**Average word = 1.3 tokens**

```
Quick formula:
Estimated tokens ≈ Word count × 1.3

Examples:
├─ 100 words → ~130 tokens
├─ 1,000 words → ~1,300 tokens
└─ 10,000 words → ~13,000 tokens
```

**Better formula (accounting for languages):**
```
Estimated tokens = (Word count × 1.3) × Language multiplier

English (1.3x):      100 words → 130 tokens
French (1.5x):       100 words → 195 tokens
Telugu (12x):        100 words → 1,560 tokens
```

### Token Counting Across Different Models

Different models have different tokenizers!

```
Same text: "The quick brown fox jumps over the lazy dog"

GPT-2 Tokenizer:   9 tokens
GPT-3/3.5 Tokenizer: 9 tokens
GPT-4 Tokenizer:    9 tokens
Claude Tokenizer:   8 tokens
LLaMA Tokenizer:    10 tokens
```

**Why different?** Each model trains its own vocabulary.

### Accounting for Hidden Tokens

**Real-world token count formula:**

```
Total Tokens = System Prompt Tokens
             + User Input Tokens
             + Chat History Tokens
             + Tool Output Tokens
             + Reserved/Buffer Tokens
             ──────────────────────
             = True Total

Example (128k context window):
├─ System prompt: 2,000 tokens
├─ Your question: 500 tokens
├─ Previous 5 messages: 30,000 tokens
├─ Retrieved documents: 50,000 tokens
├─ Buffer for response: 20,000 tokens
├─ Model overhead: 5,000 tokens
└─ TOTAL USED: 107,500 tokens
   Available for new response: 20,500 tokens
```

---

## Advanced: How Tokenizers Are Built

### The Tokenization Algorithm: Byte-Pair Encoding (BPE)

Most modern tokenizers use **Byte-Pair Encoding** (BPE). Here's how it works:

#### Step 1: Start with bytes

```
Text: "hello"
Raw bytes: [104, 101, 108, 108, 111]  (ASCII codes)
```

#### Step 2: Count frequency of adjacent pairs

```
Byte pairs in "hello":
├─ (104, 101) appears 1 time
├─ (101, 108) appears 1 time
├─ (108, 108) appears 1 time
└─ (108, 111) appears 1 time
```

#### Step 3: Merge the most frequent pair

```
Before: [104, 101, 108, 108, 111]
Most frequent pair: (104, 101) → Assign new ID: 256

After: [256, 108, 108, 111]
Vocabulary update: 256 = [104, 101] = "he"
```

#### Step 4: Repeat thousands of times

```
Iteration 1: Merge (104, 101) → 256 ("he")
Iteration 2: Merge (108, 108) → 257 ("ll")
Iteration 3: Merge (256, 108) → 258 ("hel")
Iteration 4: Merge (257, 111) → 259 ("llo")
...
After 50,000 iterations: Vocabulary complete
```

#### Final Vocabulary Size: 50,000+ tokens

```
ID | Token
---|-------
1  | ' '    (space)
2  | 'e'    (most common letter)
3  | 't'    (second most common)
...
256 | "he"
257 | "the"
...
50000 | "concatenate"
```

### SentencePiece: Alternative Approach

Google's models use **SentencePiece**, which:
- Treats spaces as explicit tokens
- Works better for non-English languages
- More consistent across languages

```
Text: "hello world"
SentencePiece tokenization:
├─ "▁hello" (space is explicit: ▁)
└─ "▁world"

Example: "நன்றி" (Tamil for "thanks")
├─ "▁நன"
├─ "றி"
(Better handling of non-Latin scripts)
```

### Vocabulary Training Process

```
1. Collect corpus of text
   └─ Usually billions of words from internet

2. Initialize vocabulary with all characters
   └─ "a", "b", "c", ..., " ", ".", etc.

3. Run BPE algorithm
   ├─ Count byte-pair frequencies
   ├─ Merge most common pairs
   └─ Repeat 50,000 times

4. Evaluate on test set
   ├─ Measure compression ratio
   ├─ Measure inference speed
   └─ Adjust if needed

5. Finalize and deploy
   └─ Fix vocabulary, use in production
```

---

## Embeddings Deep Dive

### What is an Embedding Vector?

An embedding is a **list of numbers** that represents a token's meaning in a multidimensional space.

```
Token: "king"
Embedding vector: [0.12, -0.45, 0.67, -0.23, 0.88, ..., 0.15]
                   └──────────────────────────────────────┘
                         768 dimensions (for GPT-3)

Token: "queen"
Embedding vector: [0.15, -0.48, 0.70, -0.20, 0.85, ..., 0.18]
                   └──────────────────────────────────────┘
                   Similar but different!
```

### How Embeddings Are Created

#### During Model Training

```
Process:
1. Initialize random embeddings for each token
   └─ Each token gets random 768-dimensional vector

2. Train model on next-token prediction
   ├─ "The cat sat on the" → predict "mat"
   ├─ Model learns to adjust embeddings
   └─ Embeddings that help prediction get strengthened

3. After millions of training examples
   └─ Embeddings converge to meaningful representations

Result: Semantically similar tokens have similar vectors
```

### Embedding Dimensions

Different models use different embedding sizes:

```
Model | Embedding Size | Model Size
------|---|---
GPT-2 | 768 | 1.5B parameters
GPT-3.5 | 1,280 | 175B parameters
GPT-4 | 12,288 | Unknown (>1T)
Claude | 4,096 | Unknown
LLaMA 2 | 4,096 | 13B-70B parameters
```

**Larger embeddings = More precision but slower**

### The Embedding Space Properties

#### 1. Semantic Similarity

Words with similar meanings are **close** in embedding space.

```
Word Relationships:
King ──────── Queen
 │              │
 └── Prince ────┘
     (closer to King)

Distance metrics:
distance(king, queen) = 0.2 (very similar)
distance(king, prince) = 0.3 (related)
distance(king, dog) = 0.9 (very different)

Measured using cosine similarity:
  Cosine(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)
  Range: -1 (opposite) to +1 (identical)
```

#### 2. Arithmetic Properties

Vector math works! (Kind of)

```
Classic example:
king - man + woman ≈ queen

Why it works:
├─ "king" vector = [0.5, 0.2, 0.8, ...]
├─ "man" vector = [0.4, 0.1, 0.7, ...]
├─ "woman" vector = [0.3, 0.1, 0.6, ...]
└─ Result ≈ [0.4, 0.2, 0.7, ...] ≈ "queen"

The model learned gender, status, and rank
as independent dimensions in embedding space!
```

#### 3. Clustering by Category

Words of same type cluster together:

```
Embedding Space (conceptual visualization):

              Animals                Numbers
                 ●                      ●
              ●     ●                 ●   ●
           ●           ●           ●       ●
    
    Verbs ●             ● Colors
    ●        ●                    ●
  ●            ●              ●    ●
                    ●     ●
```

### How Embeddings Are Used in the Model

```
Input Text: "The cat sat"
       ↓
Tokenization: [262, 1200, 3891]
       ↓
Embedding lookup:
├─ Token 262 → [0.1, -0.5, 0.3, ..., 0.2]  (768 dims)
├─ Token 1200 → [0.4, 0.1, -0.2, ..., 0.9]  (768 dims)
└─ Token 3891 → [0.2, -0.1, 0.6, ..., 0.3]  (768 dims)
       ↓
Embedding matrix (3 tokens × 768 dimensions):
[
  [0.1,  -0.5,  0.3,  ...,  0.2],
  [0.4,   0.1, -0.2,  ...,  0.9],
  [0.2,  -0.1,  0.6,  ...,  0.3]
]
       ↓
Process through transformer layers
├─ Self-attention (comparing tokens)
├─ Feed-forward networks
└─ Repeat 96 times (for GPT-4)
       ↓
Output: Next token prediction
```

### Embedding Visualization

Tools to visualize embeddings in 2D/3D:

```
Popular tools:
├─ t-SNE: Preserves local structure
├─ UMAP: Faster, preserves global structure
└─ PCA: Simple, fast, less accurate

Example t-SNE visualization of word embeddings:

      positive                    
         ↑                        
    great ●   excellent ●         
         ●awesome      ●wonderful  
    ─────┼──────────────────→ quality
   bad ●  │  good ●              
         terrible● ●worse         
         ↓                        
      negative                   
```

---

## Optimization Strategies

### Strategy 1: System Prompt Optimization

**Current (Bad):**
```
System Prompt (500 tokens):
"You are a helpful AI assistant. You should provide detailed 
explanations. Always be polite. Consider edge cases. Format code 
as markdown. Never provide dangerous code. Ask clarifying questions 
when needed. Be concise but thorough..."

Cost per 1000 calls: $5.00
```

**Optimized (Good):**
```
System Prompt (50 tokens):
"Helpful, concise AI. Format code as markdown."

Cost per 1000 calls: $0.50
Savings: $4.50!
```

**Advanced:**
Use **prompt caching** (OpenAI feature):
```
First request: 500 tokens (system prompt)
Next 3,599 requests: 0 tokens (cached!)

Cost: $5 for first call, then $0.15 for 3,599 more
Total savings: ~$18 vs $5000 without caching
```

### Strategy 2: Chat History Management

**Pattern 1: Sliding Window (Keep Last N Messages)**
```
Conversation history: Message 1, 2, 3, 4, 5, 6, 7...
Keep only last 3 messages:
  ├─ Message 5
  ├─ Message 6
  └─ Message 7

Tokens saved: ~70% reduction
Downside: Lose context from earlier messages
```

**Pattern 2: Summarization**
```
Old messages (100 tokens):
User: How do I use Python decorators?
Assistant: A decorator is a function that...

Summarize to (20 tokens):
"Discussed Python decorators and their use cases"

Add summary to new request:
"Context: We previously discussed Python decorators.
Now help me with async functions."

Tokens saved: ~80% reduction
Benefit: Keep semantic context
```

**Pattern 3: Hierarchical Summary**
```
Level 1: Last 5 messages (full)       → 100 tokens
Level 2: Previous 20 messages (summary) → 50 tokens
Level 3: Older messages (key points)   → 20 tokens
Total: 170 tokens (vs 500 for full history)

Savings: 66%
Context preserved: ~85%
```

### Strategy 3: Tool Output Filtering

**Before (Raw Output - 500 tokens):**
```json
{
  "status": "success",
  "data": {
    "id": 12345,
    "name": "John",
    "email": "john@example.com",
    "phone": "+1-555-0123",
    "address": "...",
    "profile": "...",
    "settings": "...",
    "history": "[large array]",
    "metadata": "...",
    ...
  },
  "timestamp": "2026-05-27T00:07:00Z",
  "requestId": "abc123",
  "version": "v2"
}
```

**After (Filtered - 50 tokens):**
```
Name: John
Email: john@example.com
Phone: +1-555-0123
```

**Savings: 90%** (450 tokens)

### Strategy 4: Output Format Optimization

**Format Comparison:**

| Format | Use Case | Token Overhead |
|--------|----------|-----------------|
| Natural language | General Q&A | 0% (baseline) |
| Markdown | Code/documentation | +5% |
| JSON | APIs/structured | +15% |
| XML | Complex nested data | +30% |
| Tables | Comparison | -10% (efficient!) |

**Recommendation:**
- Use tables for comparisons
- Use natural language for explanations
- Use JSON only when parsing programmatically

### Strategy 5: Request Batching

**Before (Multiple Requests):**
```
Request 1: Analyze sentiment (500 tokens input) → $0.005
Request 2: Extract entities (400 tokens input) → $0.004
Request 3: Summarize (300 tokens input) → $0.003
Total: 1,200 tokens, $0.012
```

**After (Single Batched Request):**
```
Request 1: 
"Analyze sentiment, extract entities, 
and summarize this text: [text]"
(800 tokens input) → $0.008

Savings: $0.004 (33% reduction!)
Plus: Faster (one API call instead of three)
```

### Strategy 6: Prompt Caching

**How it works:**
```
OpenAI Prompt Caching feature:

Request 1:
├─ System prompt: 2,000 tokens (1x cost)
├─ Large context: 50,000 tokens (1x cost)
└─ User query: 100 tokens (1x cost)
Total: 52,100 tokens @ 1x = $0.52

Request 2 (same hour, same context):
├─ System prompt: 2,000 tokens (cached! 10% cost)
├─ Large context: 50,000 tokens (cached! 10% cost)
└─ User query: 150 tokens (1x cost)
Total: 52,150 tokens @ ~0.15x = $0.08

Cost per request: $0.52 → $0.08 (85% savings!)
```

---

## Real-World Examples

### Example 1: Customer Support Chatbot

**Scenario:** Company uses AI for chat support

**Inefficient Setup:**
```
System prompt (tokens): 500
Per request:
├─ System: 500 tokens (repeated)
├─ Chat history: 2,000 tokens
├─ User question: 150 tokens
└─ Response: 300 tokens
Total per request: 2,950 tokens

1,000 requests/day:
├─ Tokens: 2,950,000
├─ Cost: ~$0.30/request × 1,000 = $300/day
└─ Annual: $109,500
```

**Optimized Setup:**
```
System prompt (tokens): 50 (shortened)
Per request:
├─ System: 50 tokens (cached)
├─ Chat history: 300 tokens (summarized)
├─ User question: 150 tokens
└─ Response: 300 tokens
Total per request: 800 tokens

1,000 requests/day:
├─ Tokens: 800,000
├─ Cost: ~$0.08/request × 1,000 = $80/day
└─ Annual: $29,200

Savings: $80,300/year (73% reduction!)
```

### Example 2: Document Analysis (RAG System)

**Scenario:** Analyze 50 documents (~100k tokens total)

**Inefficient:**
```
Per query:
├─ All 50 documents: 100,000 tokens (input)
├─ User query: 200 tokens (input)
├─ Response: 500 tokens (output)
Total: 100,700 tokens

Cost per query: ~$1.01
10 queries: $10.10

Problem: Always sending all documents!
```

**Optimized:**
```
Per query:
├─ Retrieve relevant docs (5/50): 10,000 tokens (input)
├─ User query: 200 tokens (input)
├─ Response: 500 tokens (output)
Total: 10,700 tokens

Cost per query: ~$0.11
10 queries: $1.10

Savings: 89% per query!
```

### Example 3: Code Generation for Telugu Developer

**Scenario:** Telugu speaker using code-generation API

**Without Optimization:**
```
Regular user (English):
├─ Query: "Write a Python function to..." (50 tokens)
├─ Response: Python code (200 tokens)
Total: 250 tokens
Cost: $0.0025

Telugu user (no optimization):
├─ Comments in Telugu: "తెలుగు విశ్లేషణ కోసం..."
├─ Query tokens: 450 tokens (9x more!)
├─ Response: Same Python code (200 tokens)
Total: 650 tokens
Cost: $0.0065 (2.6x more!)
```

**Optimized:**
```
Telugu user (optimized):
├─ Code comments in English only
├─ Query: "Write Python with analysis comments..." (50 tokens)
├─ Response: Python code (200 tokens)
Total: 250 tokens
Cost: $0.0025 (same as English user!)
```

---

## Interview Preparation

### Essential One-Liners

**Q: "What is a token?"**
```
A token is the smallest unit of text that an LLM processes.
It's not a word, but a "chunk" of characters. The tokenizer
breaks down input text into tokens, converts them to numeric
IDs, then to embedding vectors for the model to understand.
```

**Q: "Why not just use letters?"**
```
Processing each letter individually would be 5-10x slower and
less efficient. Tokens are the sweet spot—they're meaningful
chunks that compress common patterns while maintaining speed.
```

**Q: "How many tokens is a word?"**
```
On average, 1.3 tokens per word in English. But it varies:
common words like "the" = 1 token, rare words might be 3-4.
Non-English languages: 5-12x more tokens.
```

### Common Interview Questions & Answers

**Q1: "Explain how tokenization works internally"**
```
Perfect Answer:

Tokenization uses Byte-Pair Encoding (BPE). It starts by
treating text as bytes, then iteratively merges the most
common adjacent pairs. After 50,000 iterations, you have a
vocabulary of 50,000 tokens. At inference, the tokenizer
looks up each token in the vocabulary to get its ID, then
the model converts that ID to an embedding vector.
```

**Q2: "Why do non-English languages cost more?"**
```
Perfect Answer:

Tokenizers are trained primarily on English internet text.
For rare languages like Telugu, the model breaks down each
word into more individual token pieces. A Telugu word might
be 9 tokens while the equivalent English word is 1 token.
This causes 9x higher costs and 9x longer generation time
for non-English speakers.
```

**Q3: "How would you estimate token count quickly?"**
```
Perfect Answer:

Quick estimate: Word count × 1.3 = tokens (English)
For non-English, multiply by language factor:
- French/Spanish: 1.5x multiplier
- Russian/Arabic: 2-5x multiplier  
- Telugu/Hindi: 8-12x multiplier

For precise counts, use OpenAI's tokenizer tool or the
Python tiktoken library.
```

**Q4: "What are the three biggest token 'wastes'?"**
```
Perfect Answer:

1. System prompts repeated with every call
   (500 tokens × 1000 calls = 500k wasted tokens)

2. Chat history sent repeatedly
   (Can be 80% of context window)

3. Tool/API outputs with unnecessary fields
   (Sending entire JSON when only 2 fields needed)

Solutions: Caching, summarization, and filtering.
```

**Q5: "Explain embedding vectors"**
```
Perfect Answer:

Embeddings are high-dimensional vectors (768-12,288 dimensions)
that represent token meaning. During training, the model learns
to position semantically similar tokens close together in this
space. For example, "king" and "queen" have similar embeddings
because they share semantic properties. The model can do vector
math: king - man + woman ≈ queen.
```

**Q6: "How does context window affect costs?"**
```
Perfect Answer:

Every token in your request counts toward the context window.
When full, older tokens are dropped. Input tokens are cheaper
(1x cost), output tokens are expensive (3-5x cost). For
context windows, you must account for:
- System prompt (often repeated)
- User input
- Chat history
- Tool outputs
- Buffer for response generation

A 128k context window seems large, but fills up quickly!
```

### Quick Reference: Token Facts

| Fact | Value |
|------|-------|
| Average English word | 1.3 tokens |
| Average paragraph | 100-150 tokens |
| 1 page of text | 400-500 tokens |
| Small book | 50,000-100,000 tokens |
| Typical vocabulary size | 50,000-100,000 tokens |
| GPT-3.5 context | 4,096 tokens |
| GPT-4 Turbo context | 128,000 tokens |
| Input token cost | 1x multiplier |
| Output token cost | 3-5x multiplier |
| Telugu vs English cost ratio | 9x more expensive |
| Typical token waste | 70-80% of tokens |

### Interview Coding Challenge

**Q: "Estimate the total token cost of this system"**

```
Given:
- 1,000 daily users
- Average conversation: 10 messages
- System prompt: 200 tokens
- Chat history per request: 1,500 tokens
- User input per message: 150 tokens
- Model output per message: 300 tokens
- Context window: 128,000 tokens

Calculate:
1. Tokens per message
2. Tokens per user per day
3. Daily token cost (use $0.01/1k input, $0.03/1k output)
4. Optimization: How much could you save?
```

**Solution:**
```
Tokens per message:
├─ System: 200 tokens
├─ Chat history: 1,500 tokens
├─ Input: 150 tokens
├─ Output: 300 tokens
└─ Total: 2,150 tokens

Per user per day (10 messages):
2,150 × 10 = 21,500 tokens

Breakdown:
├─ Input: (200 + 1,500 + 150) × 10 = 17,500 tokens @ $0.01
├─ Output: 300 × 10 = 3,000 tokens @ $0.03
└─ Daily per user: $0.265

Daily for 1,000 users:
$0.265 × 1,000 = $265/day

Annual: $96,725

Optimization ideas:
├─ Use prompt caching: Save $158/day
├─ Summarize history: Save $180/day
├─ Filter outputs: Save $45/day
└─ Total optimization: -$383/day = Save $140,000/year!
```

---

## Summary: The 10 Key Takeaways

1. **Tokens are not words** – 1.3 tokens per word on average
2. **Tokenization uses BPE** – Merges common byte pairs iteratively
3. **Embeddings are vectors** – High-dimensional numbers representing meaning
4. **Context windows are finite** – Every token counts toward the limit
5. **Output tokens cost more** – 3-5x expensive than input tokens
6. **Non-English is expensive** – Telugu costs 9x more than English
7. **Hidden taxes waste tokens** – System prompts, history, tool outputs
8. **Token counting is estimable** – Word count × 1.3 is a quick estimate
9. **Optimization saves money** – 70-80% of tokens are often wasted
10. **Caching reduces cost** – 85% savings on repeated prompts

---

## The 30-Second Pitch (For Interviews)

> Tokens are the atomic units that LLMs process. A tokenizer breaks text into chunks (not words), assigns each an ID, then converts it to an embedding vector—a high-dimensional list of numbers representing meaning. Context windows are fixed (e.g., 128k tokens), and output tokens cost 3-5x more than input. Most systems waste 70-80% of tokens on system prompts, chat history, and unnecessary data. Understanding tokens is critical because non-English speakers pay 9x more, and smart optimization can reduce costs by 85%.

---

**Last Updated**: 2026  
**Level**: Beginner to Advanced  
**Use Cases**: Interview prep, cost optimization, system design
