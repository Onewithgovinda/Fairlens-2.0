
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def generate_loan_explanation(
    prediction,
    approval_probability,
    shap_factors
):
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured. Check your .env file."

    prompt = f"""
You are Fairlens, an explainable AI assistant.

Model prediction: {prediction}
Approval probability: {approval_probability:.2%}
SHAP factors: {shap_factors}

Explain these results in simple English.

Rules:
- Never change the model prediction.
- Do not invent applicant information.
- Do not claim SHAP proves causation.
- Do not guarantee loan approval.
- Explain that this is an educational prototype.

Use these headings:
1. Model result
2. Main influencing factors
3. What this means
4. Important limitations
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )

                if response.text:
                    return response.text

                return "Gemini returned an empty response."

            except Exception as exc:
                error_text = str(exc)

                if (
                    ("503" in error_text or "429" in error_text)
                    and attempt < 2
                ):
                    time.sleep(2 ** (attempt + 1))
                    continue

                raise

        return "Gemini is temporarily unavailable. Please try again."

    except Exception as exc:
        return (
            f"AI explanation unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
