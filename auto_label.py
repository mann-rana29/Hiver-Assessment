import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TEMPLATE_PATH = os.path.join(DATA_DIR, 'golden_set_template.csv')
GOLDEN_SET_PATH = os.path.join(DATA_DIR, 'golden_set.csv')

class LabelResult(BaseModel):
    intent: str = Field(description="The primary intent of the customer's message (e.g., 'battery_issue', 'update_issue', 'login_appleid_issue', 'hardware_damage', 'other').")
    auto_handle: bool = Field(description="True if the message requires a human agent (e.g., requires account details, complex troubleshooting). False if it can be auto-handled with standard advice.")
    escalate_reason: str = Field(description="Brief reason for escalation or 'N/A' if not escalated.")

def auto_label():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Template not found at {TEMPLATE_PATH}. Please run data_prep.py first.")
        return

    df = pd.read_csv(TEMPLATE_PATH)
    # limit to 150 for the golden set
    df = df.head(150).copy()
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=LabelResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data labeler. Label the following customer message for an AppleSupport dataset.\n\n{format_instructions}"),
        ("user", "Customer Message: {message}\nBrand Response (for context): {response}")
    ])
    
    chain = prompt | llm | parser
    
    print(f"Auto-labeling {len(df)} rows to create golden_set.csv...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = chain.invoke({
                "message": row['customer_message'],
                "response": row['brand_response'],
                "format_instructions": parser.get_format_instructions()
            })
            df.at[idx, 'intent'] = result.intent
            df.at[idx, 'auto_handle'] = result.auto_handle
            df.at[idx, 'escalate_reason'] = result.escalate_reason
        except Exception as e:
            df.at[idx, 'intent'] = 'unknown'
            df.at[idx, 'auto_handle'] = False
            df.at[idx, 'escalate_reason'] = f'Error: {e}'
            
    df.to_csv(GOLDEN_SET_PATH, index=False)
    print(f"Saved labeled golden set to {GOLDEN_SET_PATH}")

if __name__ == "__main__":
    auto_label()
