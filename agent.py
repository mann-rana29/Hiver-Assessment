import os
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class ClassificationResult(BaseModel):
    intent: str = Field(description="The primary intent of the customer's message (e.g., 'battery_issue', 'update_issue', 'login_appleid_issue', 'hardware_damage', 'other').")
    should_escalate: bool = Field(description="True if the message requires a human agent (e.g., requires account details, complex troubleshooting). False if it can be auto-handled with standard advice.")
    escalate_reason: str = Field(description="Brief reason for escalation or 'N/A' if not escalated.")

class AISupportAgent:
    def __init__(self):
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        # Using local HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self._setup_chains()
        
    def _setup_chains(self):
        self.parser = PydanticOutputParser(pydantic_object=ClassificationResult)
        
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert customer support triager for AppleSupport. "
                       "Analyze the customer's message, classify its intent, and decide if it needs human escalation.\n\n"
                       "{format_instructions}"),
            ("user", "Customer Message: {message}")
        ])
        
        self.classification_chain = classification_prompt | self.llm | self.parser
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and polite AppleSupport AI agent. "
                       "Draft a reply to the customer based on their message. "
                       "Use the provided historical resolutions to ground your response and match the brand tone.\n\n"
                       "Historical Resolutions:\n{context}"),
            ("user", "Customer Message: {message}")
        ])
        
        self.response_chain = response_prompt | self.llm
        
    def build_knowledge_base(self, df: pd.DataFrame):
        """
        Builds a simple RAG knowledge base from past resolved conversations.
        """
        documents = []
        for _, row in df.iterrows():
            if pd.notna(row.get('brand_response')):
                doc = Document(
                    page_content=f"Customer: {row['customer_message']}",
                    metadata={"response": str(row['brand_response'])}
                )
                documents.append(doc)
        
        if documents:
            self.vectorstore = Chroma.from_documents(documents, self.embeddings, collection_name="apple_support")
            
    def process_message(self, message: str) -> dict:
        """
        End-to-end processing of a customer message.
        """
        # Step 1: Classify
        try:
            classification = self.classification_chain.invoke({
                "message": message,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            result = {
                "intent": classification.intent,
                "should_escalate": classification.should_escalate,
                "escalate_reason": classification.escalate_reason,
                "draft_response": None
            }
        except Exception as e:
            # Fallback if parsing fails
            result = {
                "intent": "unknown",
                "should_escalate": True,
                "escalate_reason": f"Parsing failed: {str(e)}",
                "draft_response": None
            }
            return result
        
        # Step 2: Draft Response
        if not result["should_escalate"]:
            context = "No historical context available."
            if self.vectorstore:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
                docs = retriever.invoke(message)
                context_parts = [f"{d.page_content}\nResolution: {d.metadata['response']}" for d in docs]
                context = "\n\n".join(context_parts)
                
            response = self.response_chain.invoke({
                "message": message,
                "context": context
            })
            result["draft_response"] = response.content
            
        return result
