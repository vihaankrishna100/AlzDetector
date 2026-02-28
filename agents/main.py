import json
from .crew_runner import run_adgent_for_subject


def run_demo():
    
    subject_id = "941_S_1195" 
    result = run_adgent_for_subject(subject_id)
    print("=== RAW CREW OUTPUT ===")
    print(result["raw_output"])

if __name__ == "__main__":
    run_demo()
