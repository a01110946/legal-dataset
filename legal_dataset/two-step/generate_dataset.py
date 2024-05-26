# generate_dataset.py

import os
import json
from langchain_openai import ChatOpenAI
from legal_dataset_generator import LegalDatasetGenerator


# Access the OPENAI_API_KEY environment variable
api_key = os.environ.get('OPENAI_API_KEY')

# Specify the model name (e.g., "gpt-3.5-turbo", "gpt-4-turbo" or "gpt-4o")
model_name = "gpt-3.5-turbo"

# Check if the API key is set
if api_key is None:
    print("API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit()

# Set the path to your CSV file containing the legal documents
csv_file_path = "C://Users//ferna//.vscode//GitHub//MexicanLaws_PrePro_DataSet//MexicanLaws_Clean_Compiled_PrePro_DataSet.csv"
zip_folder_url = 'https://github.com/rafaeljosem/MNA-ProyectoIntegrador_EQ10/raw/main/small_dataset.zip'

# Create an instance of the LegalDatasetGenerator
llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0)
generator = LegalDatasetGenerator(llm=llm)

# Generate JSON objects from the CSV file
json_objects = generator.generate_from_source(
    source_type="url",
    source=zip_folder_url,
    max_items_per_document=2,  # Adjust the number of items per document as needed
    chunk_size=5000,
    chunk_overlap=400,
)

# Save the JSON objects to a file
with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(json_objects, f, ensure_ascii=False, indent=2)

print("JSON objects generated and saved to articles.json")