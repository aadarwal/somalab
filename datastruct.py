import os
import json

input_directory = '/Users/aadarsh/Desktop/somalab/transcripts'
output_file = 'medical_dialogue_dataset.json'

def label_dialogue(speaker, text):
    if speaker == 'Patient':
        if any(word in text.lower() for word in ['worried', 'scared', 'pain', 'anxious']):
            return 1 
        return 0  
    elif speaker == 'Doctor':
        if any(word in text.lower() for word in ['don’t worry', 'tests', 'reassure']):
            return 2  
        return 1  
    return 0  

def process_files():
    dataset = []
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            with open(os.path.join(input_directory, filename), 'r', encoding='ISO-8859-1') as file:
                lines = file.readlines()
                
                for line in lines:
                    if line.startswith('D:'):  
                        text = line.replace('D:', '').strip()
                        dataset.append({
                            "text": text,
                            "speaker": "Doctor",
                            "label": label_dialogue("Doctor", text)
                        })
                    elif line.startswith('P:'):  
                        text = line.replace('P:', '').strip()
                        dataset.append({
                            "text": text,
                            "speaker": "Patient",
                            "label": label_dialogue("Patient", text)
                        })
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(dataset, json_file, indent=4)                        

process_files()