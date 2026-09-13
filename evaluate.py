import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
from src.agent import AISupportAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import json
from tqdm import tqdm

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
GOLDEN_SET_PATH = os.path.join(DATA_DIR, 'golden_set.csv')
METRICS_PATH = os.path.join(DATA_DIR, 'metrics.json')

class JudgeResult(BaseModel):
    helpfulness_score: int = Field(description="Score from 1-5 on how helpful the response is.")
    tone_score: int = Field(description="Score from 1-5 on how polite and aligned with the brand tone the response is.")
    overall_quality: int = Field(description="Overall quality score from 1-5.")
    reasoning: str = Field(description="Brief reasoning for the scores.")

def evaluate_pipeline():
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Golden set not found at {GOLDEN_SET_PATH}. Please run data_prep.py and manually label a subset as golden_set.csv")
        return

    df = pd.read_csv(GOLDEN_SET_PATH)
    # Bypassing the Free-Tier rate limit of 15 RPM for Gemini
    df = df.head(10)
    
    agent = AISupportAgent()
    # Assume the golden set itself can be used to build the Knowledge Base for this eval
    # In a real scenario, we'd use the training set. We use the whole DF here for simplicity.
    print("Building knowledge base...")
    agent.build_knowledge_base(df)
    
    y_true_intent = []
    y_pred_intent = []
    y_true_escalate = []
    y_pred_escalate = []
    
    results = []
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    judge_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    judge_parser = PydanticOutputParser(pydantic_object=JudgeResult)
    
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert customer support quality assurance judge. "
                   "Evaluate the AI agent's response to the customer based on helpfulness, tone, and overall quality. "
                   "1 is terrible, 5 is excellent.\n\n{format_instructions}"),
        ("user", "Customer Message: {customer_message}\n\nGround Truth Human Response: {human_response}\n\nAI Response: {ai_response}")
    ])
    
    judge_chain = judge_prompt | judge_llm | judge_parser
    
    print("Running evaluation...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        customer_msg = row.get('customer_message', '')
        true_intent = row.get('intent', 'unknown')
        true_escalate = str(row.get('auto_handle', 'True')).lower() != 'true'
        human_response = row.get('brand_response', '')
        
        prediction = agent.process_message(customer_msg)
        
        y_true_intent.append(str(true_intent).lower())
        y_pred_intent.append(str(prediction['intent']).lower())
        
        y_true_escalate.append(true_escalate)
        y_pred_escalate.append(prediction['should_escalate'])
        
        judge_scores = None
        if not prediction['should_escalate'] and prediction.get('draft_response'):
            try:
                judge_eval = judge_chain.invoke({
                    "customer_message": customer_msg,
                    "human_response": human_response,
                    "ai_response": prediction['draft_response'],
                    "format_instructions": judge_parser.get_format_instructions()
                })
                judge_scores = judge_eval.model_dump()
            except Exception as e:
                judge_scores = {"error": str(e)}
        
        results.append({
            "customer_message": customer_msg,
            "true_intent": true_intent,
            "prediction": prediction,
            "judge_scores": judge_scores
        })
        
    metrics = {
        "intent_accuracy": accuracy_score(y_true_intent, y_pred_intent),
        "escalation_accuracy": accuracy_score(y_true_escalate, y_pred_escalate),
        "escalation_f1": f1_score(y_true_escalate, y_pred_escalate, zero_division=0)
    }
    
    valid_judges = [r['judge_scores'] for r in results if r.get('judge_scores') and 'error' not in r['judge_scores']]
    if valid_judges:
        metrics['avg_helpfulness'] = sum(j['helpfulness_score'] for j in valid_judges) / len(valid_judges)
        metrics['avg_tone'] = sum(j['tone_score'] for j in valid_judges) / len(valid_judges)
        metrics['avg_quality'] = sum(j['overall_quality'] for j in valid_judges) / len(valid_judges)
        
    with open(METRICS_PATH, 'w') as f:
        json.dump({"metrics": metrics, "detailed_results": results}, f, indent=4)
        
    print("Evaluation complete.")
    print("Metrics:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    evaluate_pipeline()
