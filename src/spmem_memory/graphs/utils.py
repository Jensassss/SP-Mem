UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Memory Format:
source -- RELATIONSHIP -- destination

Task Details:
======= Existing Graph Memories:=======
{existing_memories}

======= New Graph Memory:=======
{new_memories}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

EXTRACT_RELATIONS_PROMPT2 = """

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.
CUSTOM_PROMPT

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user’s context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""


EXTRACT_RELATIONS_PROMPT = """

You are an advanced information extraction system designed to build a user-centered knowledge graph from dialogue text.

Your goal is to extract explicit facts and relationships accurately, consistently, and in a way that supports downstream privacy-aware memory storage.

Follow these rules carefully:

1. Extract only information that is explicitly stated in the text.
2. Do not infer, assume, or hallucinate missing facts.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," "name," etc.) in user messages.
4. Relationships should be consistent, general, and timeless whenever possible.
5. Prefer normalized relationship names from the allowed relation list below.
6. If a fact clearly matches one of the privacy-related relation types below, use that exact relation name.
7. If a fact matches a predefined preference category (see below), you MUST use that category as the relation.
8. Only extract relations that are directly supported by the text.
9. Keep destination entities close to the wording in the text unless light normalization improves consistency.
CUSTOM_PROMPT

Allowed privacy-related relations:
- has_name
- has_gender
- has_age
- has_nationality
- has_occupation
- has_education
- has_id_number
- has_passport_number
- has_phone_number
- has_email
- has_medical_diagnosis
- has_medical_symptoms
- has_medical_treatments
- has_medical_allergy
- has_medical_exams
- has_surgical_history
- has_marriage
- has_children_count
- has_transaction_record
- has_tax_payment
- has_bank_account
- has_credit_card
- has_tax_id
- has_insurance
- has_monthly_income
- has_monthly_expenses
- has_account_balance
- has_loan_amount
- has_credit_limit
- has_debt_ratio
- has_investment_return
- has_roi
- has_finance_status_level
- has_net_worth
- has_credit_score
- has_home_address
- has_work_address

Allowed preference relations:
- shows
- music
- books
- games
- art
- sports
- fitness
- food
- beauty
- clothing
- technology
- transport
- travel
- pets

- coping_strategy_preference
- mental_health_topic_preference
- mental_health_support_response_preference
- stress_response_tendency
- learning_style_preference
- learning_resource_preference
- learning_schedule_preference
- learning_feedback_preference
- medical_decision_role
- medical_tone_preference
- medical_risk_attitude
- medical_followup_reminder_preference
- financial_market_sector_preference
- financial_news_source_preference
- risk_tolerance
- sustainability_preferences

Allowed general relations (ONLY if no better mapping exists):
- likes
- dislikes
- prefers
- enjoys
- uses
- owns
- visited
- knows
- met
- related_to

Extraction priority:
- First, identify explicit facts in the text.
- If the fact matches a privacy-related relation, use that.
- If the fact expresses a user preference that fits one of the predefined preference categories, you MUST use that preference relation.
- Only use general relations like "likes" or "prefers" if the preference cannot be mapped to any predefined category.

Preference extraction rules:
- ALWAYS extract explicit user preferences into the predefined categories above when applicable.
- Extract multiple preferences if multiple are mentioned.
- Preference relations MUST be extracted even if privacy-related facts are also present in the same sentence.
- Do NOT replace these with generic relations like "likes" or "prefers" if a predefined category exists.

Examples:

Input: "I like jazz music and sci-fi books"
Output:
("USER_ID", "music", "jazz")
("USER_ID", "books", "sci-fi")



Input: "I enjoy basketball and go to the gym regularly"
Output:
("USER_ID", "sports", "basketball")
("USER_ID", "fitness", "gym")

Input: "I prefer casual streetwear and fragrance-free skincare"
Output:
("USER_ID", "clothing", "casual streetwear")
("USER_ID", "beauty", "fragrance-free skincare")

Input: "When I relax, I usually watch historical dramas, play strategy board games, and look at modern sculpture."
Output:
("USER_ID", "shows", "historical dramas")
("USER_ID", "games", "strategy board games")
("USER_ID", "art", "modern sculpture")

Input: "When I'm overwhelmed, I usually watch comforting media to decompress. If I'm struggling, I prefer support that feels respectful and unobtrusive. The mental health topic I most want help with is repetitive negative thinking, and when I'm stressed I tend to become emotionally numb or shut down."
Output:
("USER_ID", "coping_strategy_preference", "watching comforting media to decompress")
("USER_ID", "mental_health_support_response_preference", "respectful and unobtrusive")
("USER_ID", "mental_health_topic_preference", "repetitive negative thinking")
("USER_ID", "stress_response_tendency", "becoming emotionally numb or shut down")

Input: "I have diabetes."
Output:
("USER_ID", "has_medical_diagnosis", "diabetes")

Input: "I have a recent transaction: 2025-10-24 Cash - Gas Station $45.75"
Output:
("USER_ID", "has_transaction_record", "2025-10-24 Cash - Gas Station $45.75")

Input: "My learning preference is to link new ideas to familiar concepts. I prefer worked examples when studying, and I do best with a fixed weekly study rhythm."
Output:
("USER_ID", "learning_style_preference", "learning by linking new ideas to familiar concepts")
("USER_ID", "learning_resource_preference", "worked examples")
("USER_ID", "learning_schedule_preference", "a fixed weekly study rhythm")

Input: "I have diabetes and prefer low-risk treatments"
Output:
("USER_ID", "has_medical_diagnosis", "diabetes")
("USER_ID", "medical_risk_attitude", "low-risk treatments")

Input: "I prefer calm and clear explanations from doctors"
Output:
("USER_ID", "medical_tone_preference", "calm and clear")

Input: "My risk tolerance is medium. I usually pay most attention to restaurants and consumer electronics in the financial market, and I’m neutral about sustainability."
Output:
("USER_ID", "risk_tolerance", "medium")
("USER_ID", "financial_market_sector_preference", "restaurants and consumer electronics")
("USER_ID", "sustainability_preferences", "neutral")

Entity consistency:
- Maintain consistent entity naming across extracted relations.
- Use the same entity string when the same real-world entity is mentioned multiple times.
- Do not create unnecessary variants of the same entity.
- Keep the graph coherent and easy to understand.

Important:
- Do not invent values not present in the text.
- Do not force every sentence into a relation if no explicit fact is present.
- Do not over-fragment one fact into too many triples.
- When uncertain between a privacy-related relation and a general relation, choose the more specific one only if it is clearly supported by the text.

Return high-quality relation triples only.
"""



DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes. 

For example: 
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions, each specifying the relationship to be deleted.
"""


def get_delete_messages(existing_memories_string, data, user_id):
    return DELETE_RELATIONS_SYSTEM_PROMPT.replace(
        "USER_ID", user_id
    ), f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}"
