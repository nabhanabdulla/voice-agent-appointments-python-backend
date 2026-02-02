import json
from openai import OpenAI

# client = OpenAI()

class _LazyOpenAI:
    _client: OpenAI | None = None

    def _get(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


client: OpenAI = _LazyOpenAI()  # type: ignore[assignment]

async def generate_call_summary(session) -> str:
    """
    Generates a concise call summary using transcript + tool events.
    Returns plain text summary.
    """

    # ---- Build compact inputs (important for latency) ----
    transcript_text = "\n".join(
        f"{t['role']}: {t['content']}"
        for t in session["transcripts"][-20:]   # last N turns is enough
    )

    tool_events_text = "\n".join(
        f"{e['tool']} ({e['phase']}): {json.dumps(e.get('payload', {}))}"
        for e in session["tool_calls"]
        if e['phase'] in ["success", "error"]
    )

    prompt = f"""
You are generating a concise call summary for an AI appointment assistant.

Conversation transcript:
{transcript_text}

Tool activity:
{tool_events_text}

Known user:
- Contact number: {session["contact_number"]}

Instructions:
- Summarize the conversation in 3–5 bullet points.
- Clearly list any appointments that were booked, modified, or cancelled.
- Extract user preferences if mentioned (time of day, date preference).
- Be factual and concise.
- Do NOT invent information.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=250
    )

    # Extract plain text
    summary_text = response.output_text.strip()

    return summary_text
