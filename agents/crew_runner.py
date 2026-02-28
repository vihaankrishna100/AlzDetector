from typing import Dict, Any

from crewai import Crew, Task, Process

from .agents import (
    make_gradcam_agent,
    make_shap_agent,
    make_clinical_agent,
    make_supervisor_agent,
)
from .tools import (
    base_model_tool,
    gradcam_tool,
    shap_tool,
    clinical_plausibility_tool,
    counterfactual_tool,
)


# Build orchestration of agents manually using crewai(look at alternatives later)
def build_adgent_crew():
    gradcam_agent = make_gradcam_agent()
    shap_agent = make_shap_agent()
    clinical_agent = make_clinical_agent()
    supervisor_agent = make_supervisor_agent()

    gradcam_task = Task(
        description=(
            "Run the 'Grad-CAM neuroimaging audit' tool for subject {subject_id}. "
            "Summarize whether the attention map focuses on anatomically plausible "
            "AD-related regions, and output a concise JSON+text assessment."
            "DO NOT: Give fake data, Make UP ROIs"
        ),
        agent=gradcam_agent,
        expected_output=(
            "A short paragraph explaining if Grad-CAM aligns with AD neuroanatomy "
            "and the raw JSON returned by the tool."
            "DO NOT GIVE FAKE DATA OR MADE UP OUTPUTS"
        ),
    )

    shap_task = Task(
        description=(
            "Run the 'SHAP clinical feature audit' tool for subject {subject_id}. "
            "Explain which clinical features most increased p(AD)."
        ),
        agent=shap_agent,
        expected_output=(
            "A short description of top 2–3 clinical drivers and JSON from the tool."
        ),
    )

    clinical_task = Task(
        description=(
            "Run the 'Clinical plausibility check' tool for subject {subject_id}. "
            "State whether the clinical profile is consistent with AD vs CN."
        ),
        agent=clinical_agent,
        expected_output=(
            "A paragraph describing clinical consistency and the JSON output."
        ),
    )

    supervisor_task = Task(

        description=(
            "For subject {subject_id}, first call the 'Base AD vs CN classifier' "
            "tool to get p(AD) and predicted label. Then call the 'Counterfactual "
            "robustness check' tool. You also have access to the other agents' "
            "reports in the shared context (Grad-CAM, SHAP, clinical plausibility).\n\n"
            "Integrate everything into:\n"
            "1. Final label (AD or CN)\n"
            "2. Final confidence (0–1) that this label is correct\n"
            "3. A 3–6 sentence explanation that cites:\n"
            "   - neuroimaging patterns (Grad-CAM summary),\n"
            "   - clinical drivers (SHAP summary),\n"
            "   - plausibility check,\n"
            "   - and counterfactual robustness.\n\n"
            "Be explicit about any disagreements between agents and how they affect "
            "your confidence. Output as structured JSON with keys: "

            "{'subject_id','final_label','final_confidence','explanation'}."
        ),
        agent=supervisor_agent,
        expected_output=(

            "Single JSON object summarizing final audited decision plus explanation."
        ),
    )

    crew = Crew(
        agents=[gradcam_agent, shap_agent, clinical_agent, supervisor_agent],

        tasks=[gradcam_task, shap_task, clinical_task, supervisor_task],
        process=Process.sequential,

        verbose=True,
    )
    return crew




#run script creates dictionary for the subject so that we can add timestam etc.
def run_adgent_for_subject(subject_id: str) -> Dict[str, Any]:

    crew = build_adgent_crew()
    result = crew.kickoff(inputs={"subject_id": subject_id})

    return {"raw_output": str(result)}
