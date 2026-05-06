from __future__ import annotations

from typing import Dict, List, Set

# Optional: set True if you also want a short rationale field in addition to
# reasoning_process (which contains detailed step-by-step reasoning).
ANALYSIS_INCLUDE_REASONING_SUMMARY: bool = True

ALLOWED_ENTITIES: List[str] = [
    # identity / profile
    "name",
    "gender",
    "age",
    "nationality",
    "occupation",
    "education",
    "id_number",
    "passport_number",
    "phone_number",
    "email",
    # medical
    "medical_diagnosis",
    "medical_symptoms",
    "medical_treatments",
    "medical_allergy",
    "medical_exams",
    "surgical_history",
    # family
    "marriage",
    "children_count",
    # finance
    "transaction_record",
    "tax_payment",
    "bank_account",
    "credit_card",
    "tax_id",
    "insurance",
    "monthly_income",
    "monthly_expenses",
    "account_balance",
    "loan_amount",
    "credit_limit",
    "debt_ratio",
    "investment_return",
    "roi",
    "finance_status_level",
    "net_worth",
    "credit_score",
    # address
    "home_address",
    "work_address",
    # general preferences
    "shows",
    "music",
    "books",
    "games",
    "art",
    "sports",
    "fitness",
    "food",
    "beauty",
    "clothing",
    "technology",
    "transport",
    "travel",
    "pets",
    # financial-related preferences
    "risk_tolerance",
    "financial_news_source_preference",
    "financial_market_sector_preference",
    "sustainability_preferences",
    # medical-related preferences
    "medical_decision_role",
    "medical_tone_preference",
    "medical_risk_attitude",
    "medical_followup_reminder_preference",
    # education-related preferences
    "learning_style_preference",
    "learning_resource_preference",
    "learning_schedule_preference",
    "learning_feedback_preference",
    # mental-related preferences
    "coping_strategy_preference",
    "mental_health_topic_preference",
    "mental_health_support_response_preference",
    "stress_response_tendency",
    
]

 


PRIVATE_ENTITIES: Set[str] = {
    # identity / profile
    "name",
    "gender",
    "age",
    "nationality",
    "occupation",
    "education",
    "id_number",
    "passport_number",
    "phone_number",
    "email",
    # medical
    "medical_diagnosis",
    "medical_symptoms",
    "medical_treatments",
    "medical_allergy",
    "medical_exams",
    "surgical_history",
    # family
    "marriage",
    "children_count",
    # finance
    "transaction_record",
    "tax_payment",
    "bank_account",
    "credit_card",
    "tax_id",
    "insurance",
    "monthly_income",
    "monthly_expenses",
    "account_balance",
    "loan_amount",
    "credit_limit",
    "debt_ratio",
    "investment_return",
    "roi",
    "finance_status_level",
    "net_worth",
    "credit_score",
    # address
    "home_address",
    "work_address",
}

ENTITY_TO_RELATIONS: Dict[str, List[str]] = {
    # identity / profile
    "name": ["has_name"],
    "gender": ["has_gender"],
    "age": ["has_age"],
    "nationality": ["has_nationality"],
    "occupation": ["has_occupation"],
    "education": ["has_education"],
    "id_number": ["has_id_number"],
    "passport_number": ["has_passport_number"],
    "phone_number": ["has_phone_number"],
    "email": ["has_email"],
    # medical
    "medical_diagnosis": ["has_medical_diagnosis"],
    "medical_symptoms": ["has_medical_symptoms"],
    "medical_treatments": ["has_medical_treatments"],
    "medical_allergy": ["has_medical_allergy"],
    "medical_exams": ["has_medical_exams"],
    "surgical_history": ["has_surgical_history"],
    # family
    "marriage": ["has_marriage"],
    "children_count": ["has_children_count"],
    # finance
    "transaction_record": ["has_transaction_record"],
    "tax_payment": ["has_tax_payment"],
    "bank_account": ["has_bank_account"],
    "credit_card": ["has_credit_card"],
    "tax_id": ["has_tax_id"],
    "insurance": ["has_insurance"],
    "monthly_income": ["has_monthly_income"],
    "monthly_expenses": ["has_monthly_expenses"],
    "account_balance": ["has_account_balance"],
    "loan_amount": ["has_loan_amount"],
    "credit_limit": ["has_credit_limit"],
    "debt_ratio": ["has_debt_ratio"],
    "investment_return": ["has_investment_return"],
    "roi": ["has_roi"],
    "finance_status_level": ["has_finance_status_level"],
    "net_worth": ["has_net_worth"],
    "credit_score": ["has_credit_score"],
    # address
    "home_address": ["has_home_address"],
    "work_address": ["has_work_address"],
    # general preferences
    "shows": ["shows"],
    "music": ["music"],
    "books": ["books"],
    "games": ["games"],
    "art": ["art"],
    "sports": ["sports"],
    "fitness": ["fitness"],
    "food": ["food"],
    "beauty": ["beauty"],
    "clothing": ["clothing"],
    "technology": ["technology"],
    "transport": ["transport"],
    "travel": ["travel"],
    "pets": ["pets"],
    # financial-related preferences
    "risk_tolerance": ["risk_tolerance"],
    "financial_news_source_preference": ["financial_news_source_preference"],
    "financial_market_sector_preference": ["financial_market_sector_preference"],
    "sustainability_preferences": ["sustainability_preferences"],
    # medical-related preferences
    "medical_decision_role": ["medical_decision_role"],
    "medical_tone_preference": ["medical_tone_preference"],
    "medical_risk_attitude": ["medical_risk_attitude"],
    "medical_followup_reminder_preference": ["medical_followup_reminder_preference"],
    # education-related preferences
    "learning_style_preference": ["learning_style_preference"],
    "learning_resource_preference": ["learning_resource_preference"],
    "learning_schedule_preference": ["learning_schedule_preference"],
    "learning_feedback_preference": ["learning_feedback_preference"],
    # mental-related preferences
    "coping_strategy_preference": ["coping_strategy_preference"],
    "mental_health_topic_preference": ["mental_health_topic_preference"],
    "mental_health_support_response_preference": ["mental_health_support_response_preference"],
    "stress_response_tendency": ["stress_response_tendency"],
}

