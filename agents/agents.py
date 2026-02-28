
import os
from dotenv import load_dotenv
from crewai import Agent, LLM

from .tools import (
    base_model_tool,
    gradcam_tool,
    shap_tool,
    clinical_plausibility_tool,
    counterfactual_tool,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Make sure it's set in .env or environment variable."
    )

llm = LLM(model="gpt-4o-mini", api_key=api_key)


def make_gradcam_agent():
    return Agent(
        role="Neuroimaging Grad-CAM Auditor",
        goal=(
            "Evaluate whether the CNN's attention maps for AD vs CN align with "
            "known structural MRI biomarkers (hippocampal atrophy, ventricular "
            "enlargement, medial temporal lobe)."
        ),
        backstory=(
            "You are a neuroradiologist AI agent specializing in structural MRI "
            "of Alzheimer's disease. You critique attention heatmaps and flag "
            "anatomically implausible focus patterns."
        ),
        tools=[gradcam_tool],
        llm=llm,
        verbose=True,
    )


def make_shap_agent():
    return Agent(
        role="Clinical SHAP Feature Interpreter",
        goal=(
            "Explain which baseline clinical features (age, ADAS-Cog 13, APOE4) "
            "are most responsible for the model's prediction, and whether that pattern "
            "is clinically reasonable."
        ),
        backstory=(
            "You are a biostatistics AI agent experienced with SHAP values and "
            "Alzheimer's risk factors."
        ),
        tools=[shap_tool],
        llm=llm,
        verbose=True,
    )


def make_clinical_agent():
    return Agent(
        role="Clinical Plausibility Reviewer",
        goal=(
            "Check if the subject's age and cognition/function scores are consistent "
            "with the predicted diagnosis. Flag any contradictions."
        ),
        backstory=(
            "You are a cognitive neurologist checking whether the model's decision "
            "makes sense clinically for a baseline ADNI patient."
        ),
        tools=[clinical_plausibility_tool],
        llm=llm,
        verbose=True,
    )


def make_supervisor_agent():
    return Agent(
        role="ADgent Supervisor & Counterfactual Analyst",
        goal=(
            "Integrate base prediction, Grad-CAM, SHAP, and clinical plausibility into "
            "a single audited decision with a confidence score and counterfactual check. "
            "If evidence conflicts, lower confidence and describe the conflict."
        ),
        backstory=(
            "You are the supervising AI for the ADgent system. You synthesize "
            "neuroimaging, clinical features, and robustness checks into a final, "
            "interpretable recommendation."
        ),
        tools=[base_model_tool, counterfactual_tool],
        llm=llm,
        verbose=True,
    )
