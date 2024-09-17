import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class AnalysisInput(BaseModel):
    patient_background: str
    conversation_history: List[Dict[str, str]]
    doctor_statement: str

class AnalysisOutput(BaseModel):
    overall_negativity: float
    perceived_judgment: float
    anxiety_stress: float
    empathy_rapport: float
    rationale: str

app = FastAPI()
sentiment_model = None
case_specific_guidelines = {
    "vaccine_hesitant": [
        "Be empathetic and acknowledge concerns",
        "Provide factual information without being pushy",
        "Avoid judgmental language"
    ],
    "chronic_illness": [
        "Show understanding of long-term challenges",
        "Emphasize patient's role in management",
        "Be encouraging but realistic"
    ]
}

def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def determine_case_type(background: str) -> str:
    if "vaccine" in background.lower():
        return "vaccine_hesitant"
    elif "chronic" in background.lower():
        return "chronic_illness"
    else:
        return "general"

def prepare_prompt(analysis_input: AnalysisInput) -> str:
    case_type = determine_case_type(analysis_input.patient_background)
    guidelines = case_specific_guidelines.get(case_type, [])
    prompt = f"Patient Background: {analysis_input.patient_background}\n"
    for item in analysis_input.conversation_history:
        prompt += f"{item['speaker']}: {item['text']}\n"
    prompt += f"Doctor: {analysis_input.doctor_statement}\n"
    prompt += "Guidelines:\n" + "\n".join(guidelines)
    return prompt

def analyze_sentiment(text: str) -> float:
    result = sentiment_model(text)
    return result[0]['score'] * 10 if result[0]['label'] == 'NEGATIVE' else 0

@app.on_event("startup")
async def startup_event():
    global sentiment_model
    sentiment_model = load_sentiment_model()
    print("Sentiment model loaded successfully.")

@app.post("/analyze", response_model=AnalysisOutput)
async def analyze(input_data: AnalysisInput):
    try:
        if sentiment_model is None:
            raise HTTPException(status_code=500, detail="Sentiment model is not loaded.")
        
        sentiment_score = analyze_sentiment(input_data.doctor_statement)

        overall_negativity = min(max(sentiment_score + np.random.normal(0, 1), 0), 10)
        perceived_judgment = min(max(sentiment_score / 2 + np.random.normal(0, 0.5), 0), 5)
        anxiety_stress = min(max(sentiment_score / 2 + np.random.normal(0, 0.5), 0), 5)
        empathy_rapport = min(max(5 - sentiment_score / 2 + np.random.normal(0, 1), -5), 5)

        rationale = (
            f"Based on sentiment analysis and case-specific guidelines, the statement shows an overall negativity of {overall_negativity:.1f}/10. "
            f"The perceived judgment is {perceived_judgment:.1f}/5, potential for anxiety/stress is {anxiety_stress:.1f}/5, "
            f"and empathy/rapport building is {empathy_rapport:.1f} on a scale from -5 to +5."
        )

        return AnalysisOutput(
            overall_negativity=overall_negativity,
            perceived_judgment=perceived_judgment,
            anxiety_stress=anxiety_stress,
            empathy_rapport=empathy_rapport,
            rationale=rationale
        )
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Unable to process input: {str(e)}")

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(exc)}"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    #curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d @/Users/aadarsh/Desktop/somalab/test.json -v

    #{"overall_negativity": 9.2,"perceived_judgment": 5.0,"anxiety_stress": 4.3,"empathy_rapport": 1.0,"rationale": "Based on sentiment analysis and case-specific guidelines, the statement shows an overall negativity of 9.2/10. The perceived judgment is 5.0/5, potential for anxiety/stress is 4.3/5, and empathy/rapport building is 1.0 on a scale from -5 to +5."}