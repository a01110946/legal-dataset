# dataset_generator.py

"""
Module: legal_dataset_generator.py
Description: Module for generating legal datasets from Mexican legal documents.
"""

import json
import time
import random
import unidecode

from typing import List, Dict
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm

from legal_document_loader import LegalDocumentLoader

class Instruction(BaseModel):
    instruction: str = Field(description="The instruction for the corresponding task.")
    output: str = Field(description="The output for the corresponding task.")
    context: str = Field(description="The context provided in the form of a fragment of the legal document to accomplish the task. It very important to extract the text accurately and completely, following the format of the legal document. Please ensure that the content of the article is correct and well-structured. If the article is 'Article 22 bis', make sure to include the term 'bis' in the article number. If the content of the article contains bullet points or numbering, make sure to include it exactly as it appears in the legal document.")

class Task(BaseModel):
    question_answering: List[Instruction] = Field(description="List of instruction-output pairs for the Question Answering task. This contains the instruction, output, and context for each pair.")
    summarization: List[Instruction] = Field(description="List of instruction-output pairs for the Summarization task. This contains the instruction, output, and context for each pair.")
    legal_advice_generation: List[Instruction] = Field(description="List of instruction-output pairs for the Legal Advice Generation task. This contains the instruction, output, and context for each pair.")
    legal_document_drafting: List[Instruction] = Field(description="List of instruction-output pairs for the Legal Document Drafting task. This contains the instruction, output, and context for each pair.")

class Dataset(BaseModel):
    items: List[Task]

    class Config:
        json_encoders = {
            Task: lambda v: pydantic_encoder(v),
            Instruction: lambda v: pydantic_encoder(v),
        }

class DatasetGenerator:
    """
    Class for generating legal datasets from Mexican legal documents.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the DatasetGenerator.

        :param llm: The language model to use for generating legal datasets.
        """
        self._llm = llm

    def generate_from_source(
        self,
        source_type: str,
        source: str,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate JSON objects for each article from the provided legal documents source.

        :param source_type: The type of source (url, csv, or dataframe).
        :param source: The source of legal documents (URL, file path, or DataFrame).
        :return: List of JSON objects representing legal documents.
        """
        if source_type == "url":
            legal_documents = LegalDocumentLoader.load_from_url(source)
        elif source_type == "csv":
            legal_documents = LegalDocumentLoader.load_from_csv(source)
        elif source_type == "dataframe":
            legal_documents = LegalDocumentLoader.load_from_dataframe(source)
        else:
            raise ValueError(f"Invalid source type: {source_type}")

        # print(f"These are the legal documents: {legal_documents}\n\n")
        # print(f"These are the legal documents type: {type(legal_documents)}\n---\n\n")

        return legal_documents

    def generate_from_legal_documents(
        self,
        legal_documents: List[Dict],
        downstream_tasks: List[str],
        max_items_per_document: int = 2,
        max_pairs_per_article: int = 2,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ) -> Dataset:
        """
        Generate JSON objects for each article from the provided list of legal documents.

        :param legal_documents: List of legal documents as strings.
        :param max_items_per_document: Maximum number of items to extract from each document.
        :param chunk_size: The target size of each text chunk (default: 1000).
        :param chunk_overlap: The overlap size between adjacent chunks (default: 200).
        :return: List of JSON objects representing articles.
        """
        pairs = []

        print(f"---\n\nThis is the legal documents: {legal_documents}\n")
        print(f"This is the legal documents type: {type(legal_documents)}\n")
        print(f"This is the type of the first legal document: {type(legal_documents[0])}\n")
        print(f"These are the number of legal documents: {len(legal_documents)}\n")
        print(f"These are the keys of the legal documents: {legal_documents[0].keys()}\n")

        for document in legal_documents:
            print(f"---\n\nThis is the doc: {document}\n")
            print(f"This is the doc type: {type(document)}\n")
            print(f"This is the doc title: {document['Title']}\n")
            print(f"This is a sample from the doc text: {document['Text'][700:900]}\n")
            try:
                # Attempt UTF-8 decoding
                title = document["Title"].encode('latin-1').decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, use unidecode
                # title = unidecode.unidecode(document["Title"])
                title = document["Title"].encode('latin-1').decode('utf-8', errors="replace")
            
                print(f"Warning: Used unidecode for title '{title}' (original: {document['Title']})")
                        
            #title = unidecode.unidecode(doc["Title"])  # Convert the title to ASCII
            #title = document["Title"].encode('latin-1').decode('utf-8')
            text = document["Text"]

            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = text_splitter.split_text(text)
            num_chunks = len(text_chunks)

            # Randomly select chunks based on max_items_per_document
            selected_chunk_indices = random.sample(range(num_chunks), min(max_items_per_document, num_chunks))
            selected_chunks = [text_chunks[i] for i in selected_chunk_indices]

            # items_per_chunk = max_items_per_document // num_chunks
            # num_remaining_items = max_items_per_document % num_chunks

            # pairs = []

            for chunk in selected_chunks:
                try:                  
                    tasks = ", ".join(downstream_tasks)  
                    print(f"---\n\nThis are the tasks: {tasks}\n")
                    parser = JsonOutputParser(pydantic_object=Task)
                    prompt = PromptTemplate(
                        template="""
                        Tomando en cuenta el fragmento de texto legal compartido, por favor genera dos ejemplos de pares instruction-output, 
                        para los siguientes tipos de tareas: "Question Answering (QA)", "Summarization", "Legal Advice Generation" y "Legal Document Drafting".

                        Para cada tarea, sigue las siguientes instrucciones:

                        Question Answering (QA):
                        - La instrucción debe ser una pregunta clara y específica basada en el contexto proporcionado, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe ser una respuesta directa y concisa a la pregunta, utilizando la información del contexto.
                        - El contexto debe incluir el número del artículo, el título y el texto completo del artículo o sección relevante.

                        Summarization:
                        - La instrucción debe solicitar un resumen del contenido del artículo o sección proporcionada, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe ser un resumen conciso que capture los puntos clave del contexto.
                        - El contexto debe incluir el número del artículo, el título y el texto completo del artículo o sección.

                        Legal Advice Generation:
                        - La instrucción debe solicitar un consejo legal basado en el contexto proporcionado, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe ser un consejo legal claro y relevante, considerando la información del contexto.
                        - El contexto debe incluir el número del artículo, el título y el texto completo del artículo o sección.

                        Legal Document Drafting:
                        - La instrucción debe solicitar la redacción de un documento legal basado en el contenido del artículo o sección, debe hacer referencia a los elementos clave del contexto, como número de artículo, nombre de la ley o similar.
                        - La salida debe seguir la estructura: "CLAUSULA [número de cláusula].- [nombre de la cláusula]. [contenido de la cláusula]."
                        - El contexto debe incluir el número del artículo, el título y el texto completo del artículo o sección.

                        Para el contexto, debes asignar el texto completo extraído de los artículos que elegiste al generar  la instrucción y la salida,
                        es muy importante que extraigas el texto de manera precisa y completa, comenzando con el número del artículo, libro o capítulo, 
                        seguido del título y el texto completo. Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo también exactamente como se presenta en el documento.
                        Solamente no debes incluir anotaciones de tipo "Párrafo adicionado DOF 15-08-2008" o "Artículo reformado DOF 31-01-1974".

                        El formato que deben seguir los ejemplos de tareas es el siguiente:\n\n{format_instructions}\n\n
                        
                        Contexto: {chunk}
                        """,
                        input_variables=["chunk"],
                        partial_variables={
                            "format_instructions": parser.get_format_instructions(),
                        },
                    )

                    query = f"Genera los ejemplos de pares instruction-output, con el siguiente fragmento de texto legal:\n{chunk}"
                    
                    chain = LLMChain(llm=self._llm, prompt=prompt)
                    result = chain.invoke(query)

                    generated_doc_json = result['text']
                    print(f"---\n\nThis is the generated doc json: {generated_doc_json}")
                    print(f"This is the generated doc json type: {type(generated_doc_json)}\n")

                    generated_task_dict = json.loads(generated_doc_json)

                    # Update the value of 'ley' to match the title of the legal document
                    #generated_task_dict['ley'] = title

                    print(f"This is the generated doc dict: {generated_task_dict}")
                    print(f"This is the generated doc dict type: {type(generated_task_dict)}\n")

                    # generated_doc = Article.model_validate_json(generated_doc_json)
                    # print(f"---\n\nThis is the generated doc: {generated_doc}")
                    # print(f"This is the generated doc type: {type(generated_doc)}\n")

                    generated_task = Task(**generated_task_dict)
                    pairs.append(generated_task)

                except Exception as e:
                    print(f"Failed to generate JSON objects for chunk: {e}")
                    continue

                time.sleep(1)

        print(f"---\n\nThis are all the pairs: {pairs}")

        return Dataset(items=pairs)
    