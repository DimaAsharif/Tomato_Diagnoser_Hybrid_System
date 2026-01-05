# Tomato Diagnoser Hybrid System
This is a hybrid system that uses machine learning (Random Forest) and rule-based expert systems (Clips) to diagnose a tomato through visible symptoms.

# Dataset
It was manually colected and revised from websites of ministries of agriculture from around the world.

**Note**: dataset need more revisions from multiple experts.

# Architecture of the System
1. User selects symptoms from the interface and hits "Diagnose".
2. The two modules give their response.
3. Confidence score is calculated for both responses and combined into Finale Trust Score (FTS).
4. All possible diagnoses are showen with their respective confidences alnog side treatment and prevention measures.
