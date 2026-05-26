# Complete LLM Interview Preparation Notes

**Last Updated:** 2026  
**Format:** Structured for easy saving and printing

---

## Table of Contents

1. [What are LLMs?](#what-are-llms)
2. [How LLMs Work Internally](#how-llms-work)
3. [Types of LLMs](#types-of-llms)
4. [Latest Models (2026)](#latest-models-2026)
5. [Training Data & Knowledge Cutoffs](#training-data--knowledge-cutoffs)
6. [Web Search & RAG](#web-search--rag)
7. [Function Calling & Tool Use](#function-calling--tool-use)
8. [MCP (Model Context Protocol)](#mcp-model-context-protocol)
9. [Weather API Integration](#weather-api-integration)
10. [Atlassian MCP Server (Jira/Confluence)](#atlassian-mcp-server)
11. [Complete Flow Diagrams](#complete-flow-diagrams)
12. [Quick Revision Cheat Sheet](#quick-revision-cheat-sheet)

---

## What are LLMs?

### Simple Definition (For Interview)

An **LLM (Large Language Model)** is a computer program that has read a massive amount of text (books, websites, articles) and learned how to predict the next word in a sentence.

### One Sentence Answer

> *"It's a neural network trained on massive text data to predict the next word, and by doing that repeatedly, it can generate coherent responses."*

### Example

| Input | Prediction |
|-------|-----------|
| "The sky is" | "blue" |
| "Explain how LLMs work" | Generates full answer word by word |

### Key Point for Interview

LLMs don't "think" or "understand" — they calculate probabilities. But because they've seen so much text, those guesses look intelligent.

### Real-World Examples You Use Every Day

| Service | What It's Using | Real Example |
|---------|-----------------|---------------|
| **ChatGPT / Claude** | LLM answering questions | You: "Explain quantum computing" → AI generates full explanation |
| **Google's Gmail Smart Reply** | LLM predicting next email | Gmail suggests: "Thanks for letting me know!" |
| **LinkedIn Autocomplete** | LLM predicting your message | Starts typing "Great article..." → AI suggests completions |
| **Copilot in VS Code** | LLM predicting code | Type function name → AI writes the entire function |
| **Grammarly** | LLM checking your writing | "You have a spelling error" → AI corrects it |
| **Voice Assistants (Siri/Alexa)** | LLM understanding commands | "Hey Siri, set a timer" → LLM converts speech to action |

### How It Actually Works Behind the Scenes

**When you type:** "The weather today is"

**What LLM calculates:**
- Probability: "sunny" = 45%
- Probability: "rainy" = 30%
- Probability: "cloudy" = 15%
- Probability: "excellent" = 5%
- Other words = 5%

**What it predicts:** "sunny" (highest probability)

**Then it repeats** for "The weather today is sunny and"
- Next prediction: "warm" or "clear" or "pleasant"

This is why ChatGPT seems to "think" — it's just picking the most likely next word thousands of times!

---

## How LLMs Work Internally

### The 4-Step Process

#### Step 1: Training (The "Reading" Phase)

1. Collect huge amount of text (entire internet up to a cutoff date)
2. Cut text into small pieces called **tokens** (word or part of word)
3. Hide the last word of a sentence → ask LLM to guess it
4. Compare guess to actual answer → adjust internal "knobs" (parameters)
5. Repeat billions of times

**Real Example:**
```
Training Example 1:
Input: "Paris is the capital of"
LLM guesses: "Rome"
Correct answer: "France"
Result: LLM slightly adjusts to be more likely to predict "France" next time

Training Example 2:
Input: "The water boiled because the temperature was"
LLM guesses: "cold"
Correct answer: "hot"
Result: LLM learns that "hot" is more likely after "boiled"

After billions of these corrections, LLM becomes very good at predicting!
```

**How many tokens?** GPT-5.4 was trained on ~1 trillion tokens (roughly equivalent to reading Wikipedia 1,000 times)

#### Step 2: Architecture (The "Brain Structure")

- Most modern LLMs use **Transformer architecture**
- **Key component:** Attention Mechanism
  - Helps model focus on relevant words before predicting next word
  - Example: In "She gave him her phone because his was dead" — the word "his" pays attention to "him"

**How Attention Works (Real Example):**

Sentence: "The bank manager decided to issue a check"

When predicting the next word after "issue a":

```
Word "bank" in sentence → Attention weight: 2%
  ("issue a bank" doesn't make sense)

Word "manager" in sentence → Attention weight: 5%
  ("issue a manager" doesn't make sense)

Word "check" in sentence → Attention weight: 85%
  ("issue a check" makes perfect sense!)

Other words → Attention weight: 8%
```

**Why is "check" highlighted?** Because the model learned from its training data that "issue" usually comes before "check". The attention mechanism found the most relevant word!

This is why LLMs understand context. If you had written "The bank was robbed", the attention would focus on different words instead.

#### Step 3: Inference (When You Use It)

1. You type a prompt
2. LLM converts words into numbers (vectors)
3. Passes through all its layers
4. Outputs probability for every possible next word
5. Picks one word (not always highest probability — sometimes random sampling)
6. Adds that word to input
7. Repeats until response is complete

**Real-Time Example - Asking ChatGPT:**

```
YOU: "Write a haiku about coding"

STEP 1: ChatGPT receives your prompt
STEP 2: Converts to numbers: ["Write", "a", "haiku", "about", "coding"] → [12, 4, 89, 23, 156]

STEP 3-4: Runs through 96 layers of neural network magic

STEP 5: Calculates probabilities for next word:
  - "Lines" = 0.5%
  - "Bugs" = 0.3%
  - "Loops" = 0.2%
  - "Code" = 0.04%
  - "Debugging" = 0.01%
  - ...

STEP 6: Picks "Bugs" (or sometimes picks a lower probability word for variety)
STEP 7: Updates prompt: "Write a haiku about coding Bugs"

REPEAT: Does this 50+ times until it creates a complete haiku

FINAL OUTPUT:
  Bugs haunt my code
  Functions dance in darkness  
  Coffee fuels my soul
```

**This entire process takes 2-5 seconds for a full response!**

#### Step 4: Fine-tuning (Optional Extra Training)

- After main training, train a bit more on specific data
- Example: Train on question-answer pairs to make it a chatbot
- This is what OpenAI did with GPT → ChatGPT

**Real Example - GPT to ChatGPT Transformation:**

**Before Fine-tuning (Raw GPT):**
```
User: "What's the capital of France?"
GPT output: "The capital of France. Paris is known for the Eiffel Tower, 
which was built in 1889. Many tourists visit. The French revolution..."
(Rambles off-topic!)
```

**After Fine-tuning on Q&A pairs (ChatGPT):**
```
User: "What's the capital of France?"
ChatGPT output: "Paris is the capital of France."
(Concise, focused answer!)
```

**What changed?** OpenAI showed ChatGPT thousands of Q&A examples and trained it to recognize user questions and give direct answers. Now ChatGPT "understands" the conversation format.

Other fine-tuning examples:
- **CodeGPT:** Fine-tuned on code repositories → better at coding
- **MedGPT:** Fine-tuned on medical papers → better at medical questions  
- **LawGPT:** Fine-tuned on legal documents → better at legal analysis

---

## Types of LLMs

### By Architecture

| Type | What It Does | Example Models |
|------|-------------|-----------------|
| **Autoregressive (Decoder-only)** | Predicts next word one by one. Best for generating text. | GPT series, LLaMA |
| **Autoencoding (Encoder-only)** | Sees whole sentence at once. Good for understanding (fill in blank). | BERT (Google) |
| **Encoder-Decoder** | Input → compress meaning → generate output. Best for translation. | T5, BART |

**Note for Interview:** Today's most popular LLMs (GPT, LLaMA, Gemini) are decoder-only autoregressive.

### By Availability

| Type | What It Means | Example |
|------|-------------|---------|
| **Closed-source** | Cannot see or download model. Only use via API/chat. | GPT-4, Claude, Gemini |
| **Open-weight** | Can download trained weights and run locally. | LLaMA (Meta), Mistral, Falcon |
| **Open-source** | Get code, training data, and weights. | OLMo and smaller models |

---

## Latest Models (2026)

### OpenAI (GPT Series)

| Model | Release Date | Key Features |
|-------|-------------|-------------|
| **GPT-4.5** | Feb 27, 2025 | "High EQ" model, great conversation, reduced hallucinations |
| **GPT-5.4** | March 5, 2026 | 1.1M context window, 90.5% on GPQA Diamond, 91.1% on coding (SWE-bench) |
| **GPT-5.5** | Expected 2026 | Not yet released |

**Architecture:** Decoder-only Transformer  
**Access:** Closed-source  
**Cost:** $30 input / $180 output per million tokens  
**Knowledge Cutoff:** August 2025

### Google (Gemini Series)

| Model | Release Date | Key Features |
|-------|-------------|-------------|
| **Gemini 2.5 Pro** | May 20, 2025 | 1M-2M context, "Deep Think" reasoning, multimodal |
| **Gemini 3.5 Flash** | May 19, 2026 | Default model for Gemini App, fast, cheap, low latency |
| **Gemini Omni** | May 19, 2026 | Most advanced multimodal, real-time video editing, cross-app tasks |
| **Gemini 3.5 Pro** | June 2026 | More powerful version (coming) |

**Architecture:** Mixture-of-Experts (MoE) + Native multimodal  
**Access:** Closed-source  
**Users:** 900+ million monthly active users

### Meta (LLaMA Series)

| Model | Release Date | Key Features |
|-------|-------------|-------------|
| **LLaMA 3** | 2024 | Strong open-weight model (70B/405B) |
| **LLaMA 4 Scout** | 2026 | 10 million token context window (largest ever) |
| **LLaMA 4 Maverick** | 2026 | High-end open-weight model |

**Architecture:** Decoder-only Transformer  
**Access:** Open-weight

### Other Major Models (2026)

| Model | Company | License | Best For |
|-------|---------|---------|----------|
| **Qwen3** (8B-235B) | Alibaba | Apache 2.0 | Best overall local LLM, 100+ languages |
| **gpt-oss** (20B/120B) | OpenAI | Apache 2.0 | Open-weight reasoning model (surprise release!) |
| **DeepSeek-V4** | DeepSeek | MIT | Million-token context, top coding |
| **Gemma 3** (1B-27B) | Google | Gemma Terms | Best for single GPU |
| **Devstral** (24B) | Mistral | Apache 2.0 | Local coding agent |
| **Phi-4-mini** (3.8B) | Microsoft | MIT | Runs on CPU, no GPU needed |

---

## Training Data & Knowledge Cutoffs

### The Critical Limitation (Important for Interview)

A pure LLM only knows what's in its training data. It **cannot know anything beyond its training cutoff date**.

### Example

| Question | Without Web Search | With Web Search |
|----------|-------------------|-----------------|
| "Capital of France?" | ✅ Paris (from training) | ✅ Paris |
| "Who won election yesterday?" | ❌ "I don't know" or guesses | ✅ Correct answer |

### Knowledge Cutoffs for Major Models (2026)

| Model | Knowledge Cutoff |
|-------|-----------------|
| **GPT-5.4** | August 2025 |
| **Gemini 3.5** | Varies (continuously updated) |
| **LLaMA 4** | 2025-2026 depending on version |
| **Claude 3.5** | April 2025 |
### Real-World Problem: Knowledge Cutoff Failures

**Example 1: Current Events**
```
Date: May 26, 2026
User: "Who won the World Cup in 2026?"

GPT-5.4 (cutoff: Aug 2025): "I don't have information about that."
(The 2026 World Cup hasn't happened yet in its training data)

Gemini 3.5 (has web search): "Argentina won the 2026 World Cup"
(Fresh data from web search)
```

**Example 2: Stock Prices**
```
Date: May 26, 2026
User: "What's the current Apple stock price?"

Without web search: "I don't know current prices"
(LLM trained on 2025 data, prices change daily)

With web search: "Apple stock is trading at $187.50"
(Fetches latest data from stock API)
```

**Example 3: Company News**
```
User: "Did OpenAI release GPT-5.5 yet?"

GPT-5.4 (without web search): "GPT-5.5 is expected in 2026, but I don't know if it's released"
(Trained only until August 2025)

With RAG + web search: "Yes, GPT-5.5 was released on January 15, 2026"
(Gets latest news from internet)
```

**This is why companies like ChatGPT added "Web Search" feature!**
### Interview Answer

> *"LLMs have a static knowledge base from their training data. They cannot know events after their cutoff date unless you provide external context through methods like RAG or web search."*

---

## Web Search & RAG

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of it as:
- **Retrieval** = Go fetch current information
- **Augmented** = Add that information to what LLM knows
- **Generation** = Produce final answer using both

### How Web Search Works Internally

```
Your Question
    ↓
[Step 1] Does this need fresh information?
    ↓ (YES)
[Step 2] LLM generates search queries
    ↓
[Step 3] Search API fetches current web results
    ↓
[Step 4] Results injected into prompt as "context"
    ↓
[Step 5] LLM reads context + answers
```

### Real Example - Asking ChatGPT with Web Search Enabled

**User Question:** "What major tech announcements happened this month?"

**Step 1:** ChatGPT decides: "This is about current events → I need web search"

**Step 2:** ChatGPT generates queries:
- "tech announcements May 2026"
- "major AI releases May 2026"
- "tech news this week"

**Step 3:** Bing Search API returns:
```
Result 1: "Gemini 4.0 announced May 20, 2026"
Result 2: "Meta releases new coding model LLaMA 4 Pro"
Result 3: "OpenAI fixes GPT-5.4 hallucination issue"
Result 4: "Microsoft integrates Copilot in all products"
```

**Step 4:** ChatGPT adds this to the prompt:
```
"Based on recent web search results:
- Gemini 4.0 announced May 20, 2026
- Meta releases new coding model LLaMA 4 Pro
- OpenAI fixes GPT-5.4 hallucination issue
- Microsoft integrates Copilot in all products

Please answer: What major tech announcements happened this month?"
```

**Step 5:** ChatGPT answers:
```
"This month had several major announcements:
1. Gemini 4.0 was announced with improved reasoning
2. Meta released LLaMA 4 Pro for enterprise
3. OpenAI improved reliability of GPT-5.4
4. Microsoft expanded Copilot integration"
```

**Without web search:** "I'm not sure about announcements in May 2026" ❌  
**With web search:** Accurate, current information ✅

### Web Search vs Deep Search

| Feature | Web Search | Deep Search |
|---------|-----------|------------|
| **What it does** | Single search, get results | Multi-step research process |
| **Number of searches** | Usually 1 | Multiple, iterative |
| **Reasoning** | Minimal | Deep reasoning between searches |
| **Output** | Direct answer | Research report with synthesis |

### The Agentic Deep Research Flow (2026)

```
Ask Question
    ↓
Break into sub-questions (Planning)
    ↓
Search for sub-question 1 → Read → Reflect
    ↓
Search for sub-question 2 → Read → Reflect
    ↓
Cross-validate findings from multiple sources
    ↓
Produce final answer with citations
```

### What Each Company Uses for Search

| Company | Search Backend |
|---------|-----------------|
| **OpenAI (ChatGPT)** | Proprietary + Bing partnership |
| **Google (Gemini)** | Google Search (native) |
| **Microsoft (Copilot)** | Bing Search |
| **Perplexity AI** | Mix of Bing + own crawlers |

### Important Limitations (For Interview)

- Model decides when to search — may skip and hallucinate
- Paywalled content is blocked
- Search results can be wrong → garbage in, garbage out
- User must explicitly enable web search in many clients

---

## Function Calling & Tool Use

### What is Function Calling?

A mechanism where LLMs can call external APIs to get real-time data or perform actions.

**The Technical Term:** Function Calling or Tool Use or Tool Calling

### How It Works (Step by Step)

```
Your Question: "What's the weather in Tokyo?"
    ↓
LLM decides: "I need weather data"
    ↓
LLM generates a function call: 
{"name": "weather_api", "args": {"location": "Tokyo"}}
    ↓
System calls external Weather API
    ↓
Weather API returns: {"temperature": 22, "condition": "sunny"}
    ↓
LLM takes that data and answers: "It's 22°C and sunny in Tokyo"
```

### Real-World Function Calling Examples

**Example 1: Google Assistant**
```
User: "Book a flight to Paris for next Friday"

Google Assistant:
1. Recognizes: This needs function call → "book_flight"
2. Generates: {
     "function": "book_flight",
     "args": {"destination": "Paris", "date": "next Friday"}
   }
3. System calls: Airline booking API
4. API returns: "Available flights found, cheapest is $450"
5. Assistant responds: "I found 12 flights. The cheapest is $450."
```

**Example 2: Slack Bot**
```
User: "@bot what's the status of project Alpha?"

Slack Bot:
1. Recognizes: Needs to fetch project info → "get_project_status"
2. Generates: {"function": "get_project_status", "project": "Alpha"}
3. System calls: Jira API
4. Jira returns: {"status": "In Progress", "completed": "45%", "deadline": "June 15"}
5. Bot responds: "Project Alpha is 45% complete. Deadline: June 15."
```

**Example 3: Shopping Assistant**
```
User: "Find me a red T-shirt under $50"

Assistant:
1. Recognizes: Needs to search inventory → "search_products"
2. Generates: {
     "function": "search_products",
     "color": "red",
     "category": "t-shirt",
     "max_price": 50
   }
3. System calls: Product database
4. Database returns: [{
     "name": "Classic Red Tee",
     "price": "$35",
     "stock": "In stock"
   }]
5. Assistant responds: "Found 3 red T-shirts under $50. Best match: Classic Red Tee for $35!"
```

### Tool Definition Example (What Developers Write)

```json
{
  "name": "weather_api",
  "description": "Get current weather conditions for a specific location",
  "parameters": {
    "location": {"type": "string", "description": "City name"}
  }
}
```

### What Each Company Uses

| Company | Tool Gateway |
|---------|--------------|
| **OpenAI** | Custom Function Calling API |
| **Google** | Built-in Tools |
| **Anthropic** | Tool Use API |
| **Perplexity** | Citation-focused tools |

---

## MCP (Model Context Protocol)

### What is MCP?

**MCP = Model Context Protocol** — A standardized way for LLMs to connect to any external tool through a unified interface.

### Why MCP Was Created

Instead of each company building custom integrations for every tool (weather, databases, Jira, etc.), MCP provides a standard protocol that works across all AI clients.

**The Problem Before MCP:**
```
OpenAI wants to integrate weather tool:
  → Builds custom integration for ChatGPT

Google wants the same weather tool:
  → Builds different custom integration for Gemini

Anthropic wants the same weather tool:
  → Builds yet another custom integration for Claude

Result: Weather company has to maintain 3 different integrations!
This is messy, expensive, and doesn't scale.
```

**The Solution With MCP:**
```
Weather company creates ONE MCP server:
  → ChatGPT can use it
  → Gemini can use it
  → Claude can use it
  → Any MCP-compatible AI can use it

Result: Weather company maintains ONE integration!
Much cleaner and scalable.
```

### Who Created MCP?

**Anthropic** (the company behind Claude) announced and open-sourced MCP.

### The Problem MCP Solves

| Before MCP | After MCP |
|-----------|----------|
| Every AI needed custom integration | One standard protocol works everywhere |
| Developers had to write different code for each AI | Write once, use across all MCP-compatible clients |
| Limited tool availability | Thousands of MCP servers available |

### MCP Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR AI CLIENT                        │
│         (Claude Desktop, Cursor, Windsurf)              │
└─────────────────────────────────────────────────────────┘
                            ↓
                    MCP PROTOCOL (Standard)
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    MCP SERVER                            │
│         (Weather, Jira, Database, etc.)                 │
└─────────────────────────────────────────────────────────┘
                            ↓
                    EXTERNAL API
                            ↓
┌─────────────────────────────────────────────────────────┐
│              WEATHER API / JIRA API / ETC.              │
└─────────────────────────────────────────────────────────┘
```

### MCP vs Regular API Call

| Aspect | Regular API Call | MCP |
|--------|-----------------|-----|
| **Standardization** | Each AI has its own format | Universal standard |
| **Tool discovery** | Hard-coded | Dynamic discovery |
| **Cross-platform** | Not portable | Works across all MCP clients |

---

## Weather API Integration

### Available Weather MCP Servers (2026)

| MCP Server | Data Source | API Key Needed? |
|-----------|-----------|-----------------|
| **Open-Meteo MCP** | Open-Meteo | ❌ No (free) |
| **NPM Weather Server** | OpenWeatherMap/AccuWeather | ✅ Yes |
| **Caiyun Weather** | Caiyun API | ✅ Yes |
| **XMCP Weather** | Weather service | ❌ No |

### JSON Configuration Examples

#### Option 1: Easiest (No API Key)

```json
{
  "mcpServers": {
    "weather": {
      "url": "https://weather.chukai.io/mcp"
    }
  }
}
```

#### Option 2: Local Open-Meteo (Recommended)

```json
{
  "mcpServers": {
    "open-meteo": {
      "command": "uvx",
      "args": ["chuk-mcp-open-meteo"]
    }
  }
}
```

#### Option 3: With API Key (AccuWeather)

```json
{
  "mcpServers": {
    "weather": {
      "command": "npx",
      "args": ["-y", "@timlukahorstmann/mcp-weather"],
      "env": {
        "ACCUWEATHER_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Where to Put the Config File

| AI Client | Config File Location |
|-----------|----------------------|
| **Claude Desktop (Mac)** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude Desktop (Windows)** | `%APPDATA%/Claude/claude_desktop_config.json` |
| **Cursor** | Settings → MCP → Add Global MCP Server |
| **Windsurf** | Command Palette → Windsurf MCP Configuration Panel |

### What Tools Open-Meteo MCP Provides

| Tool | What It Does |
|------|-------------|
| **get_weather_forecast** | Current + hourly + daily forecast |
| **geocode_location** | Find coordinates by city name |
| **get_air_quality** | PM2.5, PM10, AQI, pollen |
| **get_historical_weather** | Data back to 1940 |
| **get_marine_forecast** | Wave conditions |
| **Batch tools** | Query up to 1000 locations at once |

---

## Atlassian MCP Server (Jira/Confluence)

### Complete JSON Configuration

```json
{
  "mcpServers": {
    "atlassian": {
      "command": "node",
      "args": ["/absolute/path/to/atlassian-mcp/dist/index.js"],
      "env": {
        "JIRA_URL": "https://your-company.atlassian.net",
        "JIRA_USERNAME": "your.email@company.com",
        "JIRA_API_TOKEN": "your_api_token",
        "CONFLUENCE_URL": "https://your-company.atlassian.net/wiki",
        "CONFLUENCE_USERNAME": "your.email@company.com",
        "CONFLUENCE_API_TOKEN": "your_api_token"
      }
    }
  }
}
```

### Where to Get Atlassian API Token

1. Go to [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens)
2. Generate new token
3. Copy and save it (shown only once)

### JSON-RPC Messages (What AI Sends to Server)

#### Calling a tool:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_jira_issue",
    "arguments": {
      "issueKey": "PROJ-123"
    }
  }
}
```

#### Server response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Issue PROJ-123: Fix login bug - Status: In Progress"
      }
    ]
  }
}
```

---

## Complete Flow Diagrams

### Flow 1: User Question with Web Search

```
┌─────────────────────────────────────────────────────────┐
│  USER: "What's the latest GPT version?"                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 1: INTENT RECOGNITION                             │
│  LLM decides: "This needs fresh data from web"         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: FUNCTION CALL GENERATED                        │
│  {                                                      │
│    "name": "web_search",                               │
│    "args": {"query": "latest GPT version 2026"}        │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: API CALL EXECUTED                              │
│  System calls Search API (Bing/Google)                 │
│  → Fetches current web results                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: RESULTS INJECTED INTO PROMPT                   │
│  "Context from web: GPT-5.4 released March 2026"       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: ANSWER GENERATED                               │
│  LLM reads search results + answers user               │
│  "The latest GPT version is GPT-5.4"                   │
└─────────────────────────────────────────────────────────┘
```

### Flow 2: MCP Server Integration

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: WRITE JSON CONFIG                              │
│  User creates config file with MCP server details      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: MCP CLIENT LOADS CONFIG                        │
│  Claude Desktop / Cursor reads the JSON                │
│  Spawns the MCP server process                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: SERVER REGISTERS TOOLS                         │
│  "Here are my tools: get_weather, get_forecast"        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: USER ASKS THE AI                               │
│  "What's the weather in Tokyo?"                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: AI GENERATES JSON-RPC CALL                     │
│  {                                                      │
│    "method": "tools/call",                             │
│    "params": {"name": "get_weather", ...}              │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 6: MCP SERVER EXECUTES                            │
│  Calls weather API with API key                        │
│  Gets real-time weather data                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 7: AI ANSWERS USER                                │
│  "It's 22°C and sunny in Tokyo"                        │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Revision Cheat Sheet

### Key Terms (Must Know for Interview)

| Term | Quick Definition |
|------|-----------------|
| **LLM** | Large Language Model - predicts next word |
| **Token** | Small piece of text (word or subword) |
| **Transformer** | The architecture behind modern LLMs |
| **Attention** | Mechanism that focuses on relevant words |
| **Parameters** | The "knobs" adjusted during training (billions/trillions) |
| **Inference** | Using a trained model to generate responses |
| **Fine-tuning** | Additional training on specific data |
| **RAG** | Retrieval-Augmented Generation - fetching external data |
| **Function Calling** | LLM calling external APIs |
| **MCP** | Model Context Protocol - standard for tool integration |
| **Context Window** | How much text LLM can process at once |

### Latest Model Stats (2026)

| Model | Context Window | Knowledge Cutoff |
|-------|----------------|-----------------|
| **GPT-5.4** | 1.1M tokens | August 2025 |
| **Gemini 3.5** | 1-2M tokens | Continuous |
| **LLaMA 4** | 10M tokens | 2025-2026 |

### Quick Comparison Table

| Feature | GPT-5.4 | Gemini 3.5 | LLaMA 4 |
|---------|---------|-----------|---------|
| **Architecture** | Decoder-only | MoE | Decoder-only |
| **Open/Closed** | Closed | Closed | Open-weight |
| **Multimodal** | Partial | Yes (Omni) | No |
| **Best for** | Coding, reasoning | Speed, integration | Local deployment |

### Common Interview Q&A

#### Q: What is the main limitation of pure LLMs?

**A:** They cannot know anything beyond their training cutoff date without external context.

#### Q: How do LLMs get current information?

**A:** Through RAG (Retrieval-Augmented Generation) or web search, where the LLM fetches current data before answering.

#### Q: What is MCP and why is it important?

**A:** Model Context Protocol is a standard that allows LLMs to connect to any external tool through a unified interface. It solves the problem of custom integrations.

#### Q: What's the difference between web search and deep search?

**A:** Web search does one query and answers. Deep search is agentic — it plans multiple searches, reflects on results, and searches again based on what it learned.

#### Q: How does function calling work?

**A:** The LLM generates a structured JSON output specifying which tool to call and with what arguments. The system executes it and injects results back.

#### Q: What architecture do most modern LLMs use?

**A:** Decoder-only Transformer with autoregressive generation.

### The "Perfect Answer" for Interview

> *"An LLM is a neural network trained on massive text data to predict the next word. It uses a Transformer architecture with attention mechanisms. While powerful, it has a static knowledge cutoff. To get current information, it uses RAG — retrieving external data via function calling before generating responses. The industry is standardizing tool integration through MCP, which allows LLMs to connect to any external API through a unified protocol."*

---

## BONUS: Real-World Scenarios & Examples

### Scenario 1: Student Asking ChatGPT for Homework Help

**Student:** "Explain photosynthesis in simple terms"

**What ChatGPT Does:**
1. ✅ Has knowledge from training data (photosynthesis was taught in training)
2. ✅ Generates response word by word
3. ✅ Answer: "Photosynthesis is when plants use sunlight to make food..."
4. ❌ Doesn't need web search (old knowledge is fine)

**Result:** Works perfectly! ChatGPT gives a great answer.

---

### Scenario 2: Journalist Asking AI for Breaking News

**Journalist:** "What are the top news stories today?"

**What Happens:**
1. ❌ Pure LLM: "I can only tell you about 2024 events"
2. ✅ With Web Search: Fetches CNN, BBC, AP News
3. ✅ With RAG: Returns: "Breaking: Major earthquake in Mexico, Stock market rises 2%, New treaty signed"
4. ✅ With Function Calling: Can call news APIs and get structured data

**Result:** Must have web search enabled!

---

### Scenario 3: Software Developer Using Copilot with MCP

**Developer:** "What's the status of bug JIRA-456?"

**What Copilot Does with MCP:**
1. Recognizes question needs data
2. Calls MCP Atlassian Server with: `{"function": "get_issue", "key": "JIRA-456"}`
3. Atlassian MCP retrieves data from actual Jira instance
4. Returns: `{"status": "In Progress", "assigned_to": "John", "due_date": "May 28"}`
5. Copilot responds: "Bug JIRA-456 is in progress, assigned to John, due May 28"

**Result:** Real-time, accurate information directly from Jira!

---

### Scenario 4: Business Analyst Using Multiple MCP Servers

**Analyst:** "Generate a report: What's the weather in our 5 main offices, and what issues are we tracking?"

**What Happens with MCP:**
```
AI receives question
  ↓
Calls MCP Weather Server:
  → Gets weather for NYC, London, Tokyo, Sydney, Mumbai
  ↓
Calls MCP Atlassian Server:
  → Gets open Jira issues and Confluence pages
  ↓
Calls MCP Database Server:
  → Gets employee count per office
  ↓
Combines all data and generates report

Final Output:
"NYC: 22°C sunny. 15 issues open. 250 employees.
London: 18°C rainy. 8 issues open. 180 employees.
...(continues for all offices)"
```

**Result:** One AI + Multiple MCP Servers = Complete business report!

---

### Scenario 5: The Problem LLMs Still Can't Solve

**Engineer:** "Code this feature for me"

**ChatGPT (Without Tools):** Writes the code, but:
- ❌ Doesn't test it
- ❌ Doesn't know your company's API structure  
- ❌ Can't access your company database schema
- ❌ Can't commit to GitHub

**ChatGPT (With MCP Server + Function Calling):**
1. Calls `get_company_database_schema` MCP tool → Learns your database
2. Calls `get_api_documentation` → Learns your APIs
3. Calls `test_code` function → Tests the code
4. Calls `github_commit` → Pushes to GitHub

**Result:** AI becomes 10x more useful! This is the future.

---

## The Accuracy Problem & Solution

### What is "Hallucination"?

A **hallucination** is when an LLM confidently says something false.

**Real Examples:**
```
User: "What's the phone number of Google's CEO?"
Pure LLM: "Sundar Pichai's personal number is +1-555-0123" (WRONG! Made it up)

User: "Tell me about a movie called 'The Floating City' starring Tom Hanks"
Pure LLM: "It's a great sci-fi film about time travelers..." (WRONG! Doesn't exist)

User: "What's the current Bitcoin price?"
Pure LLM: "Bitcoin is $42,000 per coin" (WRONG! Could be outdated or inaccurate)
```

**Why does this happen?**

Because LLMs predict text based on patterns, not facts. They're really good at predicting what words *should* come next, but they don't verify if it's true. It's like an autocomplete that sounds confident even when wrong.

### Before 2024: Pure LLMs (Unreliable)

**The Problem:**
- ChatGPT gave wrong information but sounded correct
- No way to verify answers
- Especially bad for current events and real-time data
- Users couldn't trust it for important decisions

**Real Problem:**
```
Doctor: "Let me ask ChatGPT about this medication dosage"
ChatGPT: "The dosage is 500mg three times daily" (Sounds confident but could be wrong!)
Doctor: "That doesn't match my reference... this AI isn't reliable"
```

### Now 2026: Multi-Layered Accuracy System (Reliable)

Modern LLMs combine MULTIPLE techniques to ensure accuracy:

#### Layer 1: Web Search (Real-Time Data)

```
User: "What's today's stock price of Apple?"

Old ChatGPT: "I don't know current prices"

New ChatGPT with Web Search:
1. Recognizes: Need current data
2. Calls: Bing Search API
3. Fetches: Real stock data from Yahoo Finance
4. Answers: "Apple is trading at $189.45 today"

Result: ✅ Accurate, current information
```

#### Layer 2: RAG (Retrieval-Augmented Generation)

```
Company Scenario: Employee asks HR AI
"What's the new parental leave policy?"

Without RAG:
AI: "I think parental leave is 6 weeks" (Old training data, wrong)

With RAG:
1. System searches company documents
2. Finds: Official policy from HR system ("New policy: 12 weeks")
3. AI reads the actual document
4. Answers: "The new parental leave policy is 12 weeks"

Result: ✅ Accurate company-specific information
```

#### Layer 3: Function Calling (Real Databases)

```
Bank AI asks: "What's John's account balance?"

Without Function Calling:
AI: "I think you have $5,000" (Hallucinated guess)

With Function Calling:
1. Calls actual banking database
2. Returns: {"account_balance": 12847.50}
3. AI answers: "Your account balance is $12,847.50"

Result: ✅ Accurate live data from authoritative source
```

#### Layer 4: MCP Servers (Standardized Integration)

```
Developer asks: "Show me all P1 bugs and current weather"

Without MCP (Old way):
- Different integration for Jira
- Different integration for weather API
- Data might be inconsistent

With MCP (New way):
1. MCP Jira Server: Fetches real P1 bugs from Jira
2. MCP Weather Server: Fetches real weather data
3. Both return fresh, verified data
4. AI combines both: "3 P1 bugs open. Weather is 22°C sunny."

Result: ✅ Accurate data from multiple verified sources
```

#### Layer 5: Deep Search (Multi-Step Verification)

```
Researcher: "Is quantum computing ready for real-world use?"

Web Search (Fast, 1 query):
"Yes, it's ready" (First result says this, but incomplete answer)

Deep Search (Thorough, multiple queries):
1. Search: "quantum computing current applications 2026"
   → Find: Limited applications in specific fields
2. Search: "quantum computing challenges 2026"
   → Find: Still has error rates, scalability issues
3. Search: "quantum computing timeline 2026"
   → Find: Experts predict 5-10 years for widespread use
4. Cross-validate all sources
5. AI concludes: "Quantum computing is ready for some specific
   applications (finance, drug discovery) but not general-purpose yet."

Result: ✅ Nuanced, accurate answer with multiple perspectives
```

### How These Layers Work Together

**Real Example: A News Reporter Using AI**

```
Reporter: "Are there any vaccine news I should know about?"

Step 1: AI decides "This needs fresh data" → Enables Web Search

Step 2: Web Search finds:
- "New vaccine approved by FDA today"
- "Studies show 95% effectiveness"
- "Available at pharmacies next month"

Step 3: AI uses Function Calling to:
- Call FDA database to verify approval is real
- Call medical databases to verify effectiveness stats
- Call pharmacy API to check availability

Step 4: Deep Search (MCP) checks:
- Official FDA statement (MCP Regulatory Server)
- Medical journal citations (MCP Academic Server)
- Expert commentary (MCP News Server)

Step 5: AI Cross-validates all sources and reports:
"NEW: FDA approved XYZ vaccine today with 95% effectiveness.
Available at major pharmacies starting next month.
Sources: FDA official statement, peer-reviewed studies, pharmacy confirmations."

Result: ✅ Completely accurate, multi-source verified reporting
```

### Comparison: Old vs New Accuracy

| Question | Pure LLM (2022) | Modern LLM (2026) |
|----------|-----------------|-------------------|
| "What's Bitcoin price?" | "I'm not sure, around $40k" ❌ | Web Search: "$47,350 today" ✅ |
| "Bug status in our system?" | "I don't have access" ❌ | MCP Jira: "PROJ-123 In Progress" ✅ |
| "Latest research on AI?" | Outdated 2024 info ❌ | Deep Search: 5 recent 2026 papers ✅ |
| "What's our policy?" | Guesses wrong ❌ | RAG fetches actual document ✅ |
| "Book flight tomorrow?" | "I think prices are $200" ❌ | Function Call API: Real $289 price ✅ |

### The Accuracy Revolution (2026)

**Before:** "An LLM might be right, might be wrong. Use it for ideas, not facts."

**Now:** "An LLM with web search + RAG + function calling + MCP = Trusted for important decisions."

**Why the shift?**

1. **Web Search** = No outdated info
2. **RAG** = Access to company/private data
3. **Function Calling** = Real databases, not guesses
4. **MCP** = Standardized, reliable integrations
5. **Deep Search** = Multi-source verification

**Result:** Hallucinations drop from 30% to <1%

### Interview Answer

> *"Modern LLMs solve the hallucination problem by combining multiple accuracy layers:*
> 
> *1. Web Search for current information*
> *2. RAG for proprietary/private data*
> *3. Function Calling for live database access*
> *4. MCP for standardized integrations*
> *5. Deep Search for multi-source verification*
>
> *So yes, today's LLMs give accurate data because they don't rely on prediction alone—they fetch and verify real information from trusted sources."*

---

## Summary: Why This Matters

| Before (2022) | Now (2026) | Future (2027+) |
|---|---|---|
| LLMs could only predict text | LLMs can call tools via function calling | Everything will use MCP standard |
| No access to real-time data | Can use web search and RAG | Every company will have MCP servers |
| Limited to training knowledge | Can fetch external APIs | Seamless integration everywhere |
| "Hallucination" problems common (30%) | Can reduce with tools (<5%) | Hallucinations nearly eliminated (<1%) |

---

**End of Document**
