import os
import csv
import requests
from transformers import AutoTokenizer
from fuzzywuzzy import process
from collections import defaultdict

# API Configuration
os.environ['HUGGINGFACE_TOKEN'] = "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"
MIXTRAL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

def load_symptom_disease_data(csv_file):
    """Load and process symptom-disease relationships from CSV."""
    disease_symptoms = defaultdict(set)
    symptom_diseases = defaultdict(set)
    all_symptoms = set()
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            symptom = row['Symptom'].strip().lower()
            disease = row['Disease'].strip().lower()
            all_symptoms.add(symptom)
            disease_symptoms[disease].add(symptom)
            symptom_diseases[symptom].add(disease)
    
    return disease_symptoms, symptom_diseases, all_symptoms

def query_mistral(text):
    """Extract symptoms using Mistral AI model."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    prompt = f"""Extract only the medical symptoms from the following text. Reply with just a comma-separated list of symptoms, nothing else.
    
Text: {text}
Symptoms:"""
    
    messages = [
        {"role": "system", "content": "You are a medical symptom detector. Extract and list only the symptoms mentioned."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_text = tokenizer.decode(inputs[0])
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(MIXTRAL_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if isinstance(response_data, list) and response_data:
            symptoms = response_data[0]['generated_text'].split("[/INST]")[-1].strip()
            return [s.strip().lower() for s in symptoms.split(',') if s.strip()]
            
    except Exception as e:
        print(f"Error in Mistral analysis: {str(e)}")
    return []

def match_symptoms(detected_symptoms, all_symptoms, disease_symptoms, symptom_diseases):
    """Match detected symptoms with known symptoms and identify potential diseases."""
    matched_symptoms = set()
    potential_diseases = defaultdict(set)
    
    for symptom in detected_symptoms:
        matches = process.extract(symptom, all_symptoms, limit=1)
        for match, score in matches:
            if score >= 60:
                matched_symptoms.add(match)
                for disease in symptom_diseases[match]:
                    potential_diseases[disease].add(match)
    
    if not matched_symptoms:
        return matched_symptoms, {}
    
    final_diseases = {}
    for disease in potential_diseases:
        disease_matched_symptoms = potential_diseases[disease]
        match_percentage = (len(disease_matched_symptoms) / len(detected_symptoms)) * 100
        
        if match_percentage >= 60:
            final_diseases[disease] = {
                'matched_symptoms': disease_matched_symptoms,
                'match_percentage': match_percentage,
                'total_symptoms': len(disease_symptoms[disease])
            }
    
    return matched_symptoms, dict(sorted(
        final_diseases.items(),
        key=lambda x: x[1]['match_percentage'],
        reverse=True
    )[:3])

def ask_additional_symptoms(disease_symptoms, disease_info, confirmed_symptoms):
    """Ask questions about additional symptoms for potential diseases."""
    disease_scores = defaultdict(lambda: {'score': 0, 'total_questions': 0, 'symptoms': set()})
    
    print("\nI'll ask some additional questions to help with the diagnosis.")
    
    for disease, info in disease_info.items():
        print(f"\nChecking symptoms for: {disease.title()}")
        unconfirmed_symptoms = disease_symptoms[disease] - confirmed_symptoms
        questions_asked = 0
        disease_scores[disease]['symptoms'].update(confirmed_symptoms)
        
        for symptom in unconfirmed_symptoms:
            if questions_asked >= 5:
                break
                
            response = input(f"Do you experience {symptom}? (yes/no/unsure): ").lower().strip()
            if response == 'quit':
                return None
            
            if response == 'yes':
                disease_scores[disease]['symptoms'].add(symptom)
                disease_scores[disease]['score'] += 1
            if response in ['yes', 'no']:
                disease_scores[disease]['total_questions'] += 1
            
            questions_asked += 1
    
    final_scores = {}
    for disease, scores in disease_scores.items():
        total_answered = scores['total_questions'] + len(info['matched_symptoms'])
        if total_answered > 0:
            confidence = ((scores['score'] + len(info['matched_symptoms'])) / total_answered) * 100
            final_scores[disease] = {
                'confidence': confidence,
                'symptoms': scores['symptoms']
            }
    
    return final_scores

def analyze_with_ai(symptoms):
    """Analyze symptoms using Mixtral AI for advanced medical analysis."""
    symptoms_text = ", ".join(sorted(symptoms))
    
    prompt = f"""As a medical expert, analyze these symptoms and provide potential diagnoses:

Symptoms: {symptoms_text}

Based on these symptoms, provide your analysis in exactly this format:

CONDITION 1: [most likely condition]
CONFIDENCE 1: [percentage]
EXPLANATION 1: [brief explanation why this matches]

CONDITION 2: [second most likely condition]
CONFIDENCE 2: [percentage]

CONDITION 3: [third most likely condition]
CONFIDENCE 3: [percentage]"""

    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        messages = [
            {"role": "system", "content": "You are a medical expert. Analyze symptoms and suggest potential conditions with explanations."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_text = tokenizer.decode(inputs[0])
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(MIXTRAL_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if isinstance(response_data, list) and response_data:
            analysis = response_data[0]['generated_text'].split("[/INST]")[-1].strip()
            return parse_ai_response(analysis, symptoms)
            
    except Exception as e:
        print(f"\nNote: Advanced AI analysis unavailable: {str(e)}")
    return None

def parse_ai_response(text, symptoms):
    """Parse AI response into structured format."""
    results = []
    current_condition = {}
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith('condition'):
            if current_condition:
                results.append(current_condition)
            current_condition = {'symptoms': symptoms}
            current_condition['disease'] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('confidence'):
            try:
                conf_str = line.split(':', 1)[1].strip().rstrip('%')
                current_condition['confidence'] = float(conf_str)
            except (ValueError, IndexError):
                current_condition['confidence'] = 0
        elif line.lower().startswith('explanation'):
            current_condition['explanation'] = line.split(':', 1)[1].strip()
    
    if current_condition:
        results.append(current_condition)
    
    return results

def print_results(csv_results, ai_results=None, all_symptoms=None):
    """Print final diagnosis results showing top 2 from each analysis."""
    print("\n=== Diagnosis Results ===")
    
    if all_symptoms:
        print("\nAll Identified Symptoms:", ", ".join(sorted(all_symptoms)))
    
    if csv_results:
        print("\nBased on symptom database analysis (Top 2):")
        for disease, info in list(csv_results.items())[:2]:  # Take only top 2
            print(f"\n{disease.title()}:")
            print(f"Confidence: {info['confidence']:.1f}%")
            print("Matched symptoms:", ", ".join(info['symptoms']))
    
    if ai_results:
        print("\nBased on AI medical analysis (Top 2):")
        for result in ai_results[:2]:  # Take only top 2
            print(f"\n{result['disease'].title()}:")
            print(f"Confidence: {result['confidence']:.1f}%")
            if 'explanation' in result and result['explanation']:
                print("Explanation:", result['explanation'])
    
    print("\nIMPORTANT: This is not a medical diagnosis.")
    print("Please consult a healthcare professional for proper medical evaluation.")

def main():
    print("Welcome to the Enhanced Medical Diagnosis System")
    
    csv_file = 'new_symptom_disease_relations.csv'
    disease_symptoms, symptom_diseases, all_symptoms = load_symptom_disease_data(csv_file)
    
    while True:
        user_input = input("\nDescribe your symptoms (or type 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        # Step 1: Mistral Symptom Detection
        detected_symptoms = query_mistral(user_input)
        if not detected_symptoms:
            print("No symptoms detected. Please try describing them differently.")
            continue
            
        print("\nDetected Symptoms:")
        for symptom in detected_symptoms:
            print(f"- {symptom}")
        
        # Step 2: CSV Database Matching and Questions
        matched_symptoms, potential_diseases = match_symptoms(
            detected_symptoms, all_symptoms, disease_symptoms, symptom_diseases)
        
        all_confirmed_symptoms = set(detected_symptoms)
        
        if potential_diseases:
            print("\nAnalyzing potential conditions...")
            csv_results = ask_additional_symptoms(
                disease_symptoms, potential_diseases, matched_symptoms)
            
            if csv_results is None:  # User quit during questions
                continue
            
            # Update all confirmed symptoms
            for info in csv_results.values():
                all_confirmed_symptoms.update(info['symptoms'])
            
            # Step 3: AI Analysis with all symptoms
            print("\nPerforming advanced medical analysis...")
            ai_results = analyze_with_ai(all_confirmed_symptoms)
            
            # Print Results with all symptoms
            print_results(csv_results, ai_results, all_confirmed_symptoms)
        else:
            # No CSV matches, go directly to AI analysis
            print("\nNo matches found in database. Performing AI analysis...")
            ai_results = analyze_with_ai(all_confirmed_symptoms)
            print_results(None, ai_results, all_confirmed_symptoms)

if __name__ == "__main__":
    main()