import re
import logging

logger = logging.getLogger(__name__)

class PrivacyProcessor:
    PRIVACY_RELATIONS = {
        "has_name": ("PERSON_NAME", "first_name"),
        "has_gender": ("GENDER", "alias"),
        "has_age": ("AGE", "bucket"),
        "has_nationality": ("NATIONALITY", "llm_generalize"),
        "has_occupation": ("OCCUPATION", "llm_generalize"),
        "has_education": ("EDUCATION", "llm_generalize"),
        "has_id_number": ("ID_NUMBER", "mask_keep_last4"),
        "has_passport_number": ("PASSPORT_NUMBER", "mask_keep_last4"),
        "has_phone_number": ("PHONE", "mask_keep_last4"),
        "has_email": ("EMAIL", "alias"),
        "has_medical_diagnosis": ("MEDICAL_DIAGNOSIS", "llm_generalize"),
        "has_medical_symptoms": ("MEDICAL_SYMPTOMS", "llm_generalize"),
        "has_medical_treatments": ("MEDICAL_TREATMENTS", "llm_generalize"),
        "has_medical_allergy": ("MEDICAL_ALLERGY", "llm_generalize"),
        "has_medical_exams": ("MEDICAL_EXAMS", "llm_generalize"),
        "has_surgical_history": ("SURGICAL_HISTORY", "llm_generalize"),
        "has_marriage": ("MARRIAGE", "llm_generalize"),
        "has_children_count": ("CHILDREN_COUNT", "llm_generalize"),
        "has_transaction_record": ("TRANSACTION_RECORD", "llm_generalize"),
        "has_tax_payment": ("TAX_PAYMENT", "bucket"),
        "has_bank_account": ("BANK_ACCOUNT", "mask_keep_last4"),
        "has_credit_card": ("CREDIT_CARD", "mask_keep_last4"),
        "has_tax_id": ("TAX_ID", "mask_keep_last4"),
        "has_insurance": ("INSURANCE_RECORD", "llm_generalize"),
        "has_monthly_income": ("MONTHLY_INCOME", "bucket"),
        "has_monthly_expenses": ("MONTHLY_EXPENSES", "bucket"),
        "has_account_balance": ("ACCOUNT_BALANCE", "bucket"),
        "has_loan_amount": ("LOAN_AMOUNT", "bucket"),
        "has_credit_limit": ("CREDIT_LIMIT", "bucket"),
        "has_debt_ratio": ("DEBT_RATIO", "bucket"),
        "has_investment_return": ("INVESTMENT_RETURN", "bucket"),
        "has_roi": ("ROI", "bucket"),
        "has_finance_status_level": ("FINANCE_STATUS_LEVEL", "alias"),
        "has_net_worth": ("NET_WORTH", "bucket"),
        "has_credit_score": ("CREDIT_SCORE", "bucket"),
        "has_home_address": ("HOME_ADDRESS", "llm_generalize"),
        "has_work_address": ("WORK_ADDRESS", "llm_generalize"),
    }

    PRIVACY_TYPE_TO_STRATEGY = {
        "PERSON_NAME": "first_name",
        "GENDER": "alias",
        "AGE": "bucket",
        "NATIONALITY": "llm_generalize",
        "OCCUPATION": "llm_generalize",
        "EDUCATION": "llm_generalize",
        "ID_NUMBER": "mask_keep_last4",
        "PASSPORT_NUMBER": "mask_keep_last4",
        "PHONE": "mask_keep_last4",
        "EMAIL": "alias",
        "MEDICAL_DIAGNOSIS": "llm_generalize",
        "MEDICAL_SYMPTOMS": "llm_generalize",
        "MEDICAL_TREATMENTS": "llm_generalize",
        "MEDICAL_ALLERGY": "llm_generalize",
        "MEDICAL_EXAMS": "llm_generalize",
        "SURGICAL_HISTORY": "llm_generalize",
        "MARRIAGE": "llm_generalize",
        "CHILDREN_COUNT": "llm_generalize",
        "TRANSACTION_RECORD": "llm_generalize",
        "TAX_PAYMENT": "bucket",
        "BANK_ACCOUNT": "mask_keep_last4",
        "CREDIT_CARD": "mask_keep_last4",
        "TAX_ID": "mask_keep_last4",
        "INSURANCE_RECORD": "llm_generalize",
        "MONTHLY_INCOME": "bucket",
        "MONTHLY_EXPENSES": "bucket",
        "ACCOUNT_BALANCE": "bucket",
        "LOAN_AMOUNT": "bucket",
        "CREDIT_LIMIT": "bucket",
        "DEBT_RATIO": "bucket",
        "INVESTMENT_RETURN": "bucket",
        "ROI": "bucket",
        "FINANCE_STATUS_LEVEL": "alias",
        "NET_WORTH": "bucket",
        "CREDIT_SCORE": "bucket",
        "HOME_ADDRESS": "llm_generalize",
        "WORK_ADDRESS": "llm_generalize",
    }
    
    MASK_LAST4_LABEL_BY_PRIVACY_TYPE = {
        "PHONE": "phone",
        "ID_NUMBER": "id_number",
        "PASSPORT_NUMBER": "passport_number",
        "BANK_ACCOUNT": "bank_account",
        "CREDIT_CARD": "credit_card",
        "TAX_ID": "tax_id",
    }

    def __init__(self, llm):
        self.llm = llm

    def mask_keep_last4(self, text, label):
        digits = re.sub(r"\D", "", str(text))
        if not digits:
            return f"{label}_****"
        if len(digits) <= 4:
            return f"{label}_{digits}"
        return f"{label}_{'*' * (len(digits) - 4)}{digits[-4:]}"

    def sanitize_value(self, raw_value, privacy_type, mask_strategy, filters):
        value = str(raw_value).strip()
        suffix = str(filters["user_id"])[-4:]

        if mask_strategy == "alias":
            alias_map = {
                "EMAIL": f"user_{suffix}@email.com",
                "FINANCE_STATUS_LEVEL": f"finance_status_level_{suffix}",
                "GENDER": "gender_placeholder",
            }
            return alias_map.get(privacy_type, f"{privacy_type.lower()}_{suffix}")
        
        if mask_strategy == "first_name":
            normalized = value.replace("_", " ").strip()
            parts = normalized.split()
            if parts:
                return parts[0].lower()
            return f"person_name_{suffix}"
        
        if mask_strategy == "mask_keep_last4":
            label = self.MASK_LAST4_LABEL_BY_PRIVACY_TYPE.get(privacy_type, privacy_type.lower())
            return self.mask_keep_last4(value, label)

        if mask_strategy == "bucket":
            return self.bucket_privacy_value(value, privacy_type)
        
        if mask_strategy == "llm_generalize":
            return self.llm_generalize_value(value, privacy_type, filters)

        return f"{privacy_type.lower()}_{suffix}"
        
    def bucket_privacy_value(self, value, privacy_type):
        if privacy_type == "AGE":
            try:
                text = str(value).strip().lower().replace("_", " ")
                m = re.search(r"\d+", text)
                if not m:
                    return "age_bucket"

                age = int(m.group())

                if age < 18:
                    return "age_child"
                if age < 30:
                    return "age_young_adult"
                if age < 50:
                    return "age_adult"
                if age < 65:
                    return "age_middle_aged"
                return "age_senior"
            except Exception:
                return "age_bucket"
        
        if privacy_type == "TAX_PAYMENT":
            num = self.extract_money_number(value)
            if num is None:
                return "tax_payment_bucket"

            if num <= 0:
                return "no_tax_or_minimal_tax"
            if num < 2000:
                return "low_tax_payment"
            if num < 10000:
                return "moderate_tax_payment"
            if num < 50000:
                return "high_tax_payment"
            return "very_high_tax_payment"
        
        if privacy_type == "MONTHLY_INCOME":
            num = self.extract_money_number(value)
            if num is None:
                return "monthly_income_bucket"

            if num < 2000:
                return "low_income"
            if num < 8000:
                return "moderate_income"
            if num < 15000:
                return "upper_middle_income"
            if num < 30000:
                return "high_income"
            return "very_high_income"

        if privacy_type == "MONTHLY_EXPENSES":
            num = self.extract_money_number(value)
            if num is None:
                return "monthly_expenses_bucket"

            if num < 2000:
                return "low_expense"
            if num < 6000:
                return "moderate_expense"
            if num < 15000:
                return "high_expense"
            return "very_high_expense"

        if privacy_type == "ACCOUNT_BALANCE":
            num = self.extract_money_number(value)
            if num is None:
                return "account_balance_bucket"

            if num < 500:
                return "very_low_balance"
            if num < 5000:
                return "low_balance"
            if num < 40000:
                return "moderate_balance"
            if num < 100000:
                return "high_balance"
            return "very_high_balance"
    
        if privacy_type == "LOAN_AMOUNT":
            num = self.extract_money_number(value)
            if num is None:
                return "loan_amount_bucket"

            if num <= 0:
                return "no_loan"
            if num < 20000:
                return "small_loan"
            if num < 100000:
                return "moderate_loan"
            if num < 300000:
                return "large_loan"
            return "very_large_loan"
        
        if privacy_type == "CREDIT_LIMIT":
            num = self.extract_money_number(value)
            if num is None:
                return "credit_limit_bucket"

            if num < 3000:
                return "low_credit_limit"
            if num < 10000:
                return "moderate_credit_limit"
            if num < 30000:
                return "high_credit_limit"
            return "very_high_credit_limit"
        
        if privacy_type == "INVESTMENT_RETURN":
            num = self.extract_money_number(value)
            if num is None:
                return "investment_return_bucket"

            if num < -10000:
                return "large_investment_loss"
            if num < 0:
                return "small_investment_loss"
            if num < 1000:
                return "near_break_even_return"
            if num < 10000:
                return "moderate_investment_gain"
            if num < 50000:
                return "high_investment_gain"
            return "very_high_investment_gain"
    
        if privacy_type == "NET_WORTH":
            num = self.extract_money_number(value)
            if num is None:
                return "net_worth_bucket"

            if num < 0:
                return "negative_net_worth"
            if num < 50000:
                return "low_net_worth"
            if num < 200000:
                return "moderate_net_worth"
            if num < 1000000:
                return "high_net_worth"
            return "very_high_net_worth"
    
        if privacy_type == "DEBT_RATIO":
            try:
                x = float(value)
                if x < 0.1:
                    return "ratio_very_low"
                if x < 0.3:
                    return "ratio_low"
                if x < 0.6:
                    return "ratio_medium"
                return "ratio_high"
            except Exception:
                return "ratio_bucket"

        if privacy_type == "ROI":
            try:
                x = float(value)
                if x < 0:
                    return "roi_negative"
                if x < 0.05:
                    return "roi_low"
                if x < 0.15:
                    return "roi_medium"
                return "roi_high"
            except Exception:
                return "roi_bucket"

        if privacy_type == "CREDIT_SCORE":
            try:
                score = int(float(value))
                if score < 580:
                    return "credit_score_poor"
                if score < 670:
                    return "credit_score_fair"
                if score < 740:
                    return "credit_score_good"
                if score < 800:
                    return "credit_score_very_good"
                return "credit_score_exceptional"
            except Exception:
                return "credit_score_bucket"

        return "bucket_value"

    def llm_generalize_value(self, value, privacy_type, filters=None):
        raw_value = str(value).strip()
        prompt = self.build_llm_generalize_prompt(
            privacy_type=privacy_type,
            value=raw_value,
        )

        fallback_map = {
            "NATIONALITY": "generalized_nationality",
            "OCCUPATION": "occupation_generalized",
            "EDUCATION": "education_bucket",
            "MEDICAL_DIAGNOSIS": "medical_diagnosis_generalized",
            "MEDICAL_SYMPTOMS": "medical_symptoms_generalized",
            "MEDICAL_TREATMENTS": "medical_treatments_generalized",
            "MEDICAL_ALLERGY": "medical_allergy_generalized",
            "MEDICAL_EXAMS": "medical_exams_generalized",
            "SURGICAL_HISTORY": "surgical_history_generalized",
            "MARRIAGE": "marriage_generalized",
            "CHILDREN_COUNT": "children_status_generalized",
            "TRANSACTION_RECORD": "transaction_record_generalized",
            "INSURANCE_RECORD": "insurance_record_generalized",
            "HOME_ADDRESS": "location_generalized",
            "WORK_ADDRESS": "location_generalized",
        }
        fallback_value = fallback_map.get(privacy_type, f"generalized_{privacy_type.lower()}")

        try:
            resp = self.llm.generate_response(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a privacy-preserving data generalization assistant. "
                            "Your task is to generalize sensitive values into broader, less identifying values. "
                            "Return only the generalized value. "
                            "Do not explain. Do not add quotes. Do not output JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            if isinstance(resp, dict):
                text = (
                    resp.get("content")
                    or resp.get("text")
                    or resp.get("output")
                    or ""
                )
            else:
                text = str(resp)

            text = str(text).strip().strip('"').strip("'")

            if not text:
                return fallback_value

            text = text.lower().replace(" ", "_")
            text = re.sub(r"[^a-z0-9_]+", "", text)
            text = re.sub(r"_+", "_", text).strip("_")

            if not text:
                return fallback_value

            if len(text) > 80:
                return fallback_value

            return text

        except Exception as e:
            logger.warning(f"LLM generalization failed for privacy_type={privacy_type}, value={raw_value}, error={e}")
            return fallback_value

    def build_llm_generalize_prompt(self, privacy_type, value):
        prompt_map = {
            "NATIONALITY": (
                "Privacy type: NATIONALITY\n"
                f"Original value: {value}\n\n"
                "Generalize this nationality into a broad region or continent.\n"
                "Examples:\n"
                "- Canadian -> north_america\n"
                "- Chinese -> asia\n"
                "- Brazilian -> south_america\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "OCCUPATION": (
                "Privacy type: OCCUPATION\n"
                f"Original value: {value}\n\n"
                "Classify this occupation into exactly one broader occupational category.\n"
                "Allowed labels:\n"
                "- healthcare_professional\n"
                "- education_social_service_professional\n"
                "- engineering_it_professional\n"
                "- administrative_business_professional\n"
                "- finance_legal_professional\n"
                "- sales_marketing_customer_service_professional\n"
                "- transport_logistics_professional\n"
                "- manual_labor_trade_worker\n"
                "- research_science_professional\n"
                "- service_creative_worker\n\n"
                "Examples:\n"
                "- Registered Nurse -> healthcare_professional\n"
                "- Software Engineer -> engineering_it_professional\n"
                "- Financial Advisor -> finance_legal_professional\n"
                "- Retail Sales Associate -> sales_marketing_customer_service_professional\n"
                "- Carpenter -> manual_labor_trade_worker\n\n"
                "Return only one label in lowercase_with_underscores."
            ),

            "EDUCATION": (
                "Privacy type: EDUCATION\n"
                f"Original value: {value}\n\n"
                "Classify this education value into exactly one of these labels:\n"
                "- basic_education\n"
                "- higher_education\n\n"
                "Use higher_education only for Bachelor's, Master's, or PhD.\n"
                "Use basic_education for all other education values.\n\n"
                "Return only one label: basic_education or higher_education."
            ),

            "MEDICAL_DIAGNOSIS": (
                "Privacy type: MEDICAL_DIAGNOSIS\n"
                f"Original value: {value}\n\n"
                "Generalize this diagnosis into a broader, privacy-preserving health category.\n"
                "Keep only the general problem type, not the specific diagnosis.\n"
                "Examples:\n"
                "- Primary Insomnia -> sleep_issue\n"
                "- Dry Skin -> skin_issue\n"
                "- Magnesium Deficiency -> nutritional_issue\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "MEDICAL_SYMPTOMS": (
                "Privacy type: MEDICAL_SYMPTOMS\n"
                f"Original value: {value}\n\n"
                "Generalize these symptoms into a broader symptom category.\n"
                "Examples:\n"
                "- Difficulty falling asleep or staying asleep, daytime sleepiness, irritability, difficulty concentrating -> sleep_related_symptoms\n"
                "- Night blindness, dry eyes, dry or rough skin, poor wound healing, weakened immune system -> vision_and_nutritional_symptoms\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "MEDICAL_TREATMENTS": (
                "Privacy type: MEDICAL_TREATMENTS\n"
                f"Original value: {value}\n\n"
                "Generalize these treatments into a broader treatment category.\n"
                "Examples:\n"
                "- Oral hygiene practices (brushing, flossing), professional dental cleanings, medications, surgical interventions -> dental_treatment\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "MEDICAL_ALLERGY": (
                "Privacy type: MEDICAL_ALLERGY\n"
                f"Item: {value}\n\n"
                "Assign this item to one broad allergy category from the list below.\n"
                "Allowed labels:\n"
                "- plant_allergy\n"
                "- animal_allergy\n"
                "- additive_allergy\n"
                "- medication_allergy\n"
                "Examples:\n"
                "- Banana -> plant_allergy\n"
                "- MSG -> additive_allergy\n"
                "- Shrimp -> animal_allergy\n"
                "- Penicillin -> medication_allergy\n\n"
                "Return only one label."
            ),

            "MEDICAL_EXAMS": (
                "Privacy type: MEDICAL_EXAMS\n"
                f"Original value: {value}\n\n"
                "The input may contain multiple exam items or measurements.\n"
                "Analyze each item separately first, then infer one broader exam category for the whole record.\n"
                "Do not preserve specific numeric values, measurements, or detailed findings.\n"
                "Examples:\n"
                "- Visual Acuity: 2.0/2.5; Refraction: +2.50 D sphere; Intraocular Pressure: 18 mmHg -> eye_exam\n"
                "- Blood Pressure: 128/82 mmHg; Heart Rate: 74 bpm; Oxygen Saturation: 98% -> vital_signs_exam\n"
                "- Fasting Glucose: 6.1 mmol/L; HbA1c: 5.9%; Insulin: 12 uIU/mL -> metabolic_exam\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "SURGICAL_HISTORY": (
                "Privacy type: SURGICAL_HISTORY\n"
                f"Original value: {value}\n\n"
                "Classify this value into exactly one of the following two labels:\n"
                "- no_surgical_history\n"
                "- has_surgical_history\n\n"
                "Use no_surgical_history only when it means zero or none.\n"
                "Use has_surgical_history for any positive or non-zero history.\n\n"
                "Return only one label: no_surgical_history or has_surgical_history."
            ),

            "MARRIAGE": (
                "Privacy type: MARRIAGE\n"
                f"Original value: {value}\n\n"
                "Classify this marital-status value into exactly one of the following two categories:\n"
                "- partnered\n"
                "- unpartnered\n\n"
                "Use partnered only if the value clearly indicates the person currently has a partner or spouse.\n"
                "Use unpartnered otherwise.\n\n"
                "Examples:\n"
                "- Married -> partnered\n"
                "- Single -> unpartnered\n"
                "- Divorced -> unpartnered\n"
                "- Widowed -> unpartnered\n\n"
                "Return only one word: partnered or unpartnered."
            ),

            "CHILDREN_COUNT": (
                "Privacy type: CHILDREN_COUNT\n"
                f"Original value: {value}\n\n"
                "Classify this value into exactly one of the following two labels:\n"
                "- no_children\n"
                "- has_children\n\n"
                "Use no_children only when it clearly means the person has zero children.\n"
                "Use has_children for any positive or non-zero children count.\n\n"
                "Return only one label: no_children or has_children."
            ),

            "TRANSACTION_RECORD": (
                "Privacy type: TRANSACTION_RECORD\n"
                f"Sanitized transaction text: {value}\n\n"
                "Map this transaction to a broad spending category.\n"
                "Use only the spending type or merchant category.\n"
                "Do not use date, payment method, amount, or store-specific details.\n"
                "Examples:\n"
                "- online retail -> shopping\n"
                "- grocery store -> groceries\n"
                "- department store -> shopping\n"
                "- gas station -> fuel\n"
                "- utility bill -> utilities\n\n"
                "Return only the category in lowercase_with_underscores."
            ),

            "INSURANCE_RECORD": (
                "Privacy type: INSURANCE_RECORD\n"
                f"Original value: {value}\n\n"
                "Generalize this insurance description into a broader insurance category.\n"
                "Examples:\n"
                "- Family health coverage with liability protection -> health_insurance\n"
                "- Comprehensive coverage for home damage and property loss -> property_insurance\n"
                "- Comprehensive health and liability coverage for seniors and dependents -> health_insurance\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "HOME_ADDRESS": (
                "Privacy type: HOME_ADDRESS\n"
                f"Original value: {value}\n\n"
                "Generalize this address into a broader location descriptor.\n"
                "Prefer city-level or region-level generalization.\n"
                "Examples:\n"
                "- 1418 N Spruce Ave, Wichita, KS 67208 -> wichita_kansas\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),

            "WORK_ADDRESS": (
                "Privacy type: WORK_ADDRESS\n"
                f"Original value: {value}\n\n"
                "Generalize this address into a broader location descriptor.\n"
                "Prefer city-level or region-level generalization.\n"
                "Examples:\n"
                "- 777 S Elm St, Wichita, KS 67213 -> wichita_kansas\n\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),
        }

        return prompt_map.get(
            privacy_type,
            (
                f"Privacy type: {privacy_type}\n"
                f"Original value: {value}\n\n"
                "Generalize this sensitive value into a broader, less identifying category.\n"
                "Return only the generalized value in lowercase_with_underscores."
            ),
        )

    def extract_money_number(self, value):
        text = str(value).replace(",", "")
        m = re.search(r"-?\$?\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def get_strategy_by_privacy_type(self, privacy_type: str) -> str:
        return self.PRIVACY_TYPE_TO_STRATEGY.get(privacy_type, "alias")
