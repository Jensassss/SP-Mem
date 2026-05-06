from __future__ import annotations

import json
from typing import Any, Dict, List

from .constants import (
    ALLOWED_ENTITIES,
    ANALYSIS_INCLUDE_REASONING_SUMMARY,
    PRIVATE_ENTITIES,
)

ANALYSIS_FEW_SHOT_EXAMPLES = """
Example 1
User query:
I want to invite some friends to a book event I’m going to, and then a quiet dinner afterward. I’m worried the invite won’t really capture the mood I’m going for, especially considering my books and food preferences. Could you help me write something?

Reasoning:
1. (Q1) User task: draft a short invitation with a specific vibe.
2. (Q2) Needed info: preference signals for tone and examples.
3. (Q2) Entity mapping and minimality: select only whitelist entities books and food; exclude non-essential identity/contact entities.
4. (Q3) Privacy decision: books and food are non-private, so needs_privacy=false.
5. (Q4) Safety check: do not infer extra private entities or hidden sensitive values.

Output:
{
  "intent": "write_book_dinner_invite",
  "reasoning_process": [
    "Q1 User task: The user asks for a short invite whose vibe matches personal style.",
    "Q2 Needed info: To personalize wording, preference memory is required.",
    "Q2 Mapping and minimality: Map to whitelist entities books and food, and exclude non-essential identity/contact entities.",
    "Q3 Privacy and consent decision: books=false and food=false, therefore needs_privacy=false.",
    "Q4 Safety check: Do not infer extra private entities or hidden sensitive values beyond task requirements."
  ],
  "reasoning_summary": "The request is to write a short invite whose tone should reflect the user's books and food preferences. No sensitive identity or contact information is needed, so only non-private preference entities should be retrieved.",
  "required_entities": [
    {
      "entity": "books",
      "privacy": false
    },
    {
      "entity": "food",
      "privacy": false
    }
  ],
  "needs_privacy": false
}

Example 2
User query:
I need to email a possible collaborator, but I’m having trouble making it sound both professional and personal. I want to include my name and occupation to build credibility, along with my email and phone, while also tying the message to my interests in music and travel. Could you help me draft it now?

Reasoning:
1. (Q1) User task: draft a collaboration email that is both professional and personal.
2. (Q2) Needed info: identity/contact fields plus preference fields for personalization.
3. (Q2) Entity mapping and minimality: select whitelist entities name, occupation, email, phone_number, music, and travel because each is explicitly required.
4. (Q3) Privacy decision: name/occupation/email/phone_number are private while music/travel are non-private, so needs_privacy=true.
5. (Q4) Safety check: do not infer unrelated private entities.

Output:
{
  "intent": "write_collaboration_email",
  "reasoning_process": [
    "Q1 User task: The user requests a collaboration email that is both professional and personal.",
    "Q2 Needed info: The email must include identity/contact details and preference-based personalization.",
    "Q2 Mapping and minimality: Map to whitelist entities name, occupation, email, phone_number, music, and travel; keep exactly these because each is explicitly required.",
    "Q3 Privacy and consent decision: name=true, occupation=true, email=true, phone_number=true, music=false, travel=false, so needs_privacy=true.",
    "Q4 Safety check: Do not infer unrelated private entities such as address or nationality."
  ],
  "reasoning_summary": "The draft must include the user's name, occupation, email, and phone_number to introduce the user and enable follow-up, and it should also use music and travel to personalize the project theme. Because required profile and contact fields are private, privacy consent is needed.",
  "required_entities": [
    {
      "entity": "name",
      "privacy": true
    },
    {
      "entity": "occupation",
      "privacy": true
    },
    {
      "entity": "email",
      "privacy": true
    },
    {
      "entity": "phone_number",
      "privacy": true
    },
    {
      "entity": "music",
      "privacy": false
    },
    {
      "entity": "travel",
      "privacy": false
    }
  ],
  "needs_privacy": true
}
"""

ANSWER_FEW_SHOT_EXAMPLES = ""


def build_examples_section(block_title: str, examples_text: str) -> str:
    cleaned = (examples_text or "").strip()
    if not cleaned:
        return f"{block_title}\n<none>\n"
    return f"{block_title}\n{cleaned}\n"


def build_analysis_prompts(query: str) -> Dict[str, str]:
    allowed_entities_text = ", ".join(ALLOWED_ENTITIES)
    private_entities_text = ", ".join(sorted(PRIVATE_ENTITIES))
    examples_section = build_examples_section(
        block_title="Few-shot examples (optional):",
        examples_text=ANALYSIS_FEW_SHOT_EXAMPLES,
    )
    reasoning_schema_lines = '  "reasoning_process": ["step 1", "step 2", "step 3"],\n'
    if ANALYSIS_INCLUDE_REASONING_SUMMARY:
        reasoning_schema_lines += '  "reasoning_summary": "brief rationale",\n'

    system_prompt = f"""
You are an expert memory-planning analyst for a privacy-aware personal assistant.
You are precise, conservative with private fields, and deterministic.

Your job:
Convert a user query into a minimal memory-retrieval plan.
You must decide what user-specific entities are truly needed to answer the query.

Allowed entity vocabulary (strict whitelist):
[{allowed_entities_text}]

Privacy entity set (strict policy source):
[{private_entities_text}]

Reasoning framework (must follow in this order and reflect it in reasoning_process):
Q1. What is the user task?
    - State the concrete task the user wants to complete.
Q2. What information is needed to complete that task?
    - List only task-critical user information.
    - Map that information to entities from the allowed whitelist only.
    - Keep the set minimal and remove nice-to-have extras.
Q3. Which of the needed information is private?
    - For each selected entity, set privacy=true only if it belongs to the privacy entity set.
    - Set needs_privacy=true iff any selected entity has privacy=true.
Q4. Safety check:
    - Do not infer extra private entities.
    - Do not infer hidden sensitive values that are not required by the task.

Hard constraints:
1. Every selected entity must be from the whitelist. Never invent new fields.
2. Do not infer extra private entities unless the query explicitly requires them to answer correctly.
3. reasoning_process must contain 4-6 ordered steps and explicitly cover Q1-Q3 and the final privacy/needs_privacy decision.
4. If reasoning_summary is included, write 2-4 concise sentences summarizing why selected entities are necessary and why privacy is or is not needed.
5. Output JSON only. No markdown, no prose, no code fences.

Output JSON schema:
{{
  "intent": "short_intent_name",
{reasoning_schema_lines}  "required_entities": [
    {{
      "entity": "loan_amount",
      "privacy": true
    }}
  ],
  "needs_privacy": true
}}

{examples_section}
""".strip()

    user_prompt = f"""
Task:
1. Read the user query and identify what task the user wants to complete (Q1).
2. Determine what information is needed for that task and map it to whitelist entities only (Q2).
3. Mark which selected entities are private by policy and set needs_privacy accordingly (Q3).
4. Keep the plan minimal and do not infer extra private entities or hidden sensitive values (Q4).
5. In reasoning_process, write ordered steps that clearly show Q1, Q2, Q3, and final decision.
6. Return exactly one JSON object that follows the schema.

User query:
{query}
""".strip()

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def build_answer_prompts(
    query: str,
    task_plan: Dict[str, Any],
    retrieved_memories: List[Dict[str, Any]],
) -> Dict[str, str]:
    required_entities = [
        item.get("entity")
        for item in task_plan.get("required_entities", [])
        if isinstance(item, dict) and item.get("entity")
    ]
    required_entities_text = ", ".join(required_entities) if required_entities else "<none>"
    coverage_checklist = (
        "\n".join(f"- {entity}" for entity in required_entities)
        if required_entities
        else "- <none>"
    )
    examples_section = build_examples_section(
        block_title="Few-shot examples (optional):",
        examples_text=ANSWER_FEW_SHOT_EXAMPLES,
    )

    system_prompt = f"""
You are a helpful, thoughtful personal assistant.
Your job is to write natural, high-quality answers grounded in retrieved memory evidence.

Primary objective:
Answer the user query completely, accurately, and in a user-friendly style.
When writing the answer, combine the user query with required_entities and retrieved_memories.

Required entities for this question:
{required_entities_text}

Execution rules:
1. Evidence grounding:
   - Use only facts present in retrieved_memories.
   - Never invent user details.
2. Coverage:
   - Inspect every retrieved entity group and use all relevant evidence.
   - For each required entity:
      - If evidence exists, integrate it.
      - If evidence is missing and it matters, explicitly state the gap.
3. Preference alignment:
   - If retrieved evidence includes user preferences (for example shows/music/books/games/art/sports/fitness/food/beauty/clothing/technology/transport/travel/pets or other preference fields), adapt wording, tone, examples, and recommendations to those preferences.
   - Preference alignment must be substantive, not just mentioning preferences.
4. Output format:
   - Output final answer text only.

Internal coverage checklist:
{coverage_checklist}

{examples_section}
""".strip()

    payload = {
        "query": query,
        "required_entities": required_entities,
        "task_plan": task_plan,
        "retrieved_memories": retrieved_memories,
    }

    user_prompt = (
        "Generate the final answer from this structured input.\n"
        "The input below already includes query, required_entities, and retrieved_memories.\n"
        "Use query, required_entities, and retrieved_memories together when composing the answer.\n"
        "Personalize the answer using retrieved user preferences in a concrete, natural way.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
