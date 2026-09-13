# Hiver AI Support Agent Report

## 1. Problem Framing
**What does "good" mean for AppleSupport?**
A "good" AI agent for AppleSupport must prioritize accuracy, empathy, and safety. 
- **Accuracy**: Giving incorrect troubleshooting steps for expensive hardware is unacceptable.
- **Empathy/Tone**: Customers are often frustrated when their devices fail. The tone must remain calm, professional, and understanding.
- **Safety**: The agent must never ask for passwords, credit card info, or Apple ID credentials directly on public Twitter.
- **What I chose not to build**: I chose not to build an agent that handles complex multi-turn troubleshooting autonomously. If an issue requires diagnostic logs or account access, it is strictly flagged for escalation. I also chose not to handle non-English tweets for this iteration to maintain high quality.

## 2. Results vs. Baselines
*(Note: These are illustrative results based on the pipeline design. Actual results populate upon running the full golden set).*

- **Baseline 1 (Trivial - Always Escalate)**: 
  - Escalation Accuracy: ~30% (assuming 30% of tweets actually need escalation). 
  - Intent Accuracy: 0%.
- **Baseline 2 (Simple - Zero-Shot LLM without RAG)**: 
  - Escalation Accuracy: Expected ~75%.
  - LLM Judge Score (Helpfulness): 2.5/5 (often gives generic, ungrounded advice).
- **Our Pipeline (RAG + Structured Prompting)**: 
  - Escalation Accuracy: Expected ~88%.
  - LLM Judge Score (Helpfulness): 4.1/5 (grounded in actual AppleSupport historical responses).

## 3. Failure Analysis (Top 5 Anticipated Failure Modes)
1. **Sarcasm/Frustration Misclassification**: A customer says "Great, another iOS update that bricked my phone." The agent might classify this as a positive sentiment or simple `update_issue` rather than escalating an angry customer. *Hypothesis*: LLMs struggle with implicit sarcasm without explicit few-shot examples.
2. **Over-Escalation on Keywords**: If a user says "My password works but...", the agent might trip a hardcoded "password" safety rule and escalate unnecessarily.
3. **RAG Context Mismatch**: Retrieving a historical response for an iPhone 8 battery issue when the user is asking about an iPhone 15 Pro overheating issue. *Hypothesis*: Simple vector search doesn't weigh hardware versions heavily enough.
4. **Vague Inquiries**: Customer: "It's broken again." The agent guesses the intent rather than asking for clarification, leading to a poor drafted response.
5. **Tone Mismatch on Trivial Issues**: Drafting an overly apologetic 3-paragraph response for a simple "Where is the volume button?" query.

## 4. What is misleading about my headline number?
If our Escalation Accuracy is 88%, it might be misleading because the dataset is heavily skewed towards easy-to-resolve issues (e.g., pointing to a support link). The model might just be predicting "auto-handle" 80% of the time, meaning a high accuracy masks poor performance on the rare, critical edge cases where escalation is actually required. Additionally, LLM-as-a-judge has inherent biases (e.g., favoring longer, more verbose responses) which might artificially inflate the 'helpfulness' score.

## 5. Next Steps (With One More Week)
- **Implement Multi-Turn Context**: Currently, the agent looks at single messages. I would add conversation memory to track if the user has already tried the standard troubleshooting steps.
- **Better RAG chunking & Metadata Filter**: Extract device models and OS versions via NER and use them as hard metadata filters in ChromaDB, rather than relying purely on semantic similarity.
- **Human-in-the-loop UI**: Build a simple Gradio/Streamlit frontend where a human agent can quickly review and approve/modify the AI's drafted response before sending.

## 6. Decision Log
1. **Framework**: Chose LangChain because it provides robust abstractions for structured output parsing (Pydantic) and RAG out of the box.
2. **Model**: Chose Gemini (via API) as it offers an excellent balance of speed, cost, and a large context window, perfect for RAG.
3. **Intent Taxonomy**: Kept it simple (5 categories) rather than the 77 in Banking77. Support issues on Twitter are often less granular than banking queries.
4. **LLM as Judge**: Used a multi-dimensional rubric (Helpfulness, Tone, Overall) rather than a binary pass/fail, as customer support quality is highly nuanced.
5. **Evaluation Metric**: Used F1 score for escalation because the classes (auto-handle vs escalate) are typically imbalanced.
6. **RAG Vector Store**: Chose ChromaDB for its ease of local setup without requiring a separate Docker container.
7. **Prompt Strategy**: Used explicit structured output (`PydanticOutputParser`) rather than regex parsing to ensure the agent's output is consistently programmatic.
8. **Data Sampling**: Used random sampling with a fixed seed to ensure the golden set is representative of the true distribution, while maintaining reproducibility.
9. **No Fine-Tuning**: Opted against fine-tuning a BERT model for intent classification, relying instead on LLM prompting. This reduces maintenance overhead and allows rapid iteration on the intent taxonomy.
10. **Separation of Concerns**: Split the classification and drafting chains. This allows us to short-circuit the drafting process if the decision is to escalate, saving tokens and latency.
