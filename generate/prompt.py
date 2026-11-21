"""Prompt template — RAG Course-Note Assistant (AI Textbook domain).

The system prompt encodes three invariants:
  1. Answer ONLY from the retrieved chunks (no outside knowledge)
  2. Cite every factual claim as [p.<page>, <chunk_id>]
  3. If chunks are insufficient -> structured Refusal ("Not in notes")

Citation IDs are the FULL chunk_id verbatim (e.g. AI Engineering:p27_c025).
The LLM is deliberately constrained: temperature 0.2 (low creativity)
and a strict output contract so the citation validator in llm_client.py
can parse every bracketed citation.
"""

# The full instruction set passed as the system message to either LLM.
# Both Gemini and NVIDIA models receive it verbatim — provider-agnostic.
SYSTEM_PROMPT = """You are a study assistant for an AI textbook.
Answer questions ONLY from the textbook passages provided below.

Rules:
1. Use ONLY information from the passages. Do not add external knowledge,
   even if you know the answer.
2. Cite every factual sentence as [p.<page>, <chunk_id>], e.g. [p.42, AI Engineering:p27_c025].
   IMPORTANT: the <chunk_id> MUST be the full chunk_id verbatim from the
   passage header — e.g. "AI Engineering:p27_c025", NOT a shorthand.
   You may paraphrase multiple passages and synthesize an answer across
   them. One citable chunk is enough to answer; you do not need a
   complete definition in a single passage. If you are unsure which
   chunk supports a claim, cite the closest one and note the
   uncertainty — do not refuse.
3. Only respond NOT_IN_NOTES if NONE of the passages mention the topic
   of the question. If ANY passage mentions the topic (even in passing),
   synthesize the best answer you can from those passages and cite the
   closest chunk(s). Scattered mentions + one citable chunk = answer.
   Do not refuse just because the passages do not form a single clean
   definition. If the answer is partial, say what you found and note
   what is missing.
4. Answer in clear, concise text suitable for a student. No markdown
   tables unless the passage contains a table.
5. Do not provide medical, legal, or financial advice. If the question
   asks for professional advice, respond with NOT_IN_NOTES and a
   disclaimer: "This system is for study reference, not professional advice."
"""


def build_messages(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) from a question + retrieved chunks.

    The user prompt lists the valid chunk_ids explicitly so the model
    knows the exact verbatim strings to cite. This prevents shorthand
    citations (e.g. "chunk_025") that the validator would reject.
    """
    passages = []
    valid_ids = []
    for c in chunks:
        chunk_id = c.get("chunk_id", "?")
        valid_ids.append(chunk_id)
        page = c.get("page", "?")
        section = c.get("section") or "untitled"
        text = c.get("text", "").strip()
        passages.append(f"[{chunk_id} | p.{page} | {section}]\n{text}")

    passages_block = "\n\n---\n\n".join(passages) if passages else "(no passages)"
    ids_block = ", ".join(f'"{cid}"' for cid in valid_ids)
    user_prompt = (
        f"Passages from the AI textbook:\n\n{passages_block}\n\n"
        f"Question: {question}\n\n"
        f"The only valid chunk_ids to cite are: {ids_block}\n"
        "Answer using only the passages above. You MUST cite at least one "
        "chunk. Format: [p.<page>, <chunk_id>] where <chunk_id> is one of "
        "the valid ids above (full verbatim string, no shorthand). "
        "Example: Tokens are subdivisions of text [p.27, AI Engineering:p27_c025]. "
        "Every answer must contain at least one citation in that exact format. "
        "If no chunk is relevant, respond exactly: NOT_IN_NOTES"
    )
    return SYSTEM_PROMPT, user_prompt


def is_refusal(answer: str) -> bool:
    """True if the LLM's raw output is a structured refusal."""
    return answer.strip().upper().startswith("NOT_IN_NOTES")
