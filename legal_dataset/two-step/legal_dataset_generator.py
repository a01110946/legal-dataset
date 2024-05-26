# legal_dataset_generator.py

"""
Module: legal_dataset_generator.py
Description: Module for generating legal datasets from Mexican legal documents.
"""

import json
import random
import time
from typing import Dict, List

import unidecode
from downstream_task_generator import DownstreamTaskGenerator
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from legal_document_loader import LegalDocumentLoader
from llm.openai import chat_completion_request
from pydantic import BaseModel, Field
from tqdm import tqdm
from utf8_encoder import convert_to_utf8


class Article(BaseModel):
    ley: str = Field(
        description="Se refiere al nombre de la ley, por ejemplo, 'Código Federal de Procedimientos Civiles'")
    libro_no: str = Field(
        description="Se refiere al número de libro al que pertenece, por ejemplo, 'LIBRO SEGUNDO'")
    libro_nombre: str = Field(
        description="Se refiere al nombre del libro, por ejemplo, 'De los Bienes'")
    titulo_no: str = Field(
        description="Se refiere al número de título, por ejemplo, 'TITULO SEGUNDO'")
    titulo_nombre: str = Field(
        description="Se refiere al nombre del título, por ejemplo, 'De la Sucesión por Testamento'")
    capitulo_no: str = Field(
        description="Se refiere al número de capítulo (solo si aplica, en caso de no pertenecer a un capítulo, declarar 'N/A'), por ejemplo, 'CAPITULO I'")
    capitulo_nombre: str = Field(
        description="Se refiere al nombre del capítulo, por ejemplo, 'De los Testamentos en General'")
    articulo_no: str = Field(
        description="Se refiere al número de artículo, por ejemplo, 'Artículo 1413'")
    articulo_contenido: str = Field(
        description="Se refiere al contenido del artículo, por ejemplo, 'Queda también sin efecto el legado, si el testador enajena la cosa legada; pero vale si la recobra por un título legal.'")


class LegalDatasetGenerator:
    """
    Class for generating legal datasets from Mexican legal documents.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the LegalDatasetGenerator.

        :param llm: The language model to use for generating legal datasets.
        """
        self._llm = llm

    def generate_from_source(
        self,
        source_type: str,
        source: str,
        max_items_per_document: int,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate JSON objects for each article from the provided legal documents source.

        :param source_type: The type of source (url, csv, or dataframe).
        :param source: The source of legal documents (URL, file path, or DataFrame).
        :param max_items_per_document: Maximum number of items to extract from each document.
        :param chunk_size: The target size of each text chunk (default: 1000).
        :param chunk_overlap: The overlap size between adjacent chunks (default: 200).
        :return: List of JSON objects representing articles.
        """
        if source_type == "url":
            legal_documents = LegalDocumentLoader.load_from_url(source)
        elif source_type == "csv":
            legal_documents = LegalDocumentLoader.load_from_csv(source)
        elif source_type == "dataframe":
            legal_documents = LegalDocumentLoader.load_from_dataframe(source)
        else:
            raise ValueError(f"Invalid source type: {source_type}")

        print(f"These are the legal documents: {legal_documents}\n\n")
        print(
            f"These are the legal documents type: {type(legal_documents)}\n---\n\n")

        json_objects = []
        for doc in legal_documents:
            # title = doc["Title"]
            # text = doc["Text"]
            json_obj = self.generate_from_legal_documents(
                [doc],
                max_items_per_document,
                chunk_size,
                chunk_overlap,
                **kwargs,
            )
            # json_obj[0]["Title"] = title
            json_objects.extend(json_obj)

        return json_objects

    def generate_from_legal_documents(
        self,
        legal_documents: List,
        max_items_per_document: int,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate JSON objects for each article from the provided list of legal documents.

        :param legal_documents: List of legal documents as strings.
        :param max_items_per_document: Maximum number of items to extract from each document.
        :param chunk_size: The target size of each text chunk (default: 1000).
        :param chunk_overlap: The overlap size between adjacent chunks (default: 200).
        :return: List of JSON objects representing articles.
        """
        legal_docs = []

        print(f"---\n\nThis is the legal documents: {legal_documents}\n")
        print(f"This is the legal documents type: {type(legal_documents)}\n")
        print(
            f"This is the type of the first legal document: {type(legal_documents[0])}\n")
        print(
            f"These are the number of legal documents: {len(legal_documents)}\n")
        print(
            f"These are the keys of the legal documents: {legal_documents[0].keys()}\n")

        for document in legal_documents:
            print(f"---\n\nThis is the doc: {document}\n")
            print(f"This is the doc type: {type(document)}\n")
            print(f"This is the doc title: {document['Title']}\n")
            print(
                f"This is a sample from the doc text: {document['Text'][700:900]}\n")

            title = convert_to_utf8(document["Title"].encode('latin-1'))
            # try:
            #     # Attempt UTF-8 decoding
            #     # document["Title"].encode('latin-1').decode('utf-8')
            #     title = convert_to_utf8(document["Title"])
            # except UnicodeDecodeError:
            #     # If UTF-8 fails, use unidecode
            #     # title = unidecode.unidecode(document["Title"])
            #     title = document["Title"].encode(
            #         'latin-1').decode('utf-8', errors="replace")

            #     print(
            #         f"Warning: Used unidecode for title '{title}' (original: {document['Title']})")

            # title = unidecode.unidecode(doc["Title"])  # Convert the title to ASCII
            # title = document["Title"].encode('latin-1').decode('utf-8')
            text = document["Text"]

            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = text_splitter.split_text(text)
            num_chunks = len(text_chunks)

            # Randomly select chunks based on max_items_per_document
            selected_chunk_indices = random.sample(
                range(num_chunks), min(max_items_per_document, num_chunks))
            selected_chunks = [text_chunks[i] for i in selected_chunk_indices]

            # items_per_chunk = max_items_per_document // num_chunks
            # num_remaining_items = max_items_per_document % num_chunks

            articles = []

            for chunk in selected_chunks:
                try:
                    parser = JsonOutputParser(pydantic_object=Article)
                    prompt = PromptTemplate(
                        template="""
                        Genera una tarjeta informativa de un artículo extraído del siguiente fragmento de texto legal, utilizando el esquema proporcionado.
                        Es muy importante que extraigas el texto de manera precisa y completa, siguiendo el formato de la ley.
                        Por favor, asegúrate de que el contenido del artículo sea correcto y esté bien estructurado.
                        Si el artículo es 'Artículo 22 bis', asegúrate de incluir el término 'bis' en el número del artículo.
                        Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo en el campo 'articulo_contenido'.
                        
                        Recuerda que el formato del artículo debe seguir el siguiente esquema:
                        :\n\n
                        {format_instructions}\n\n
                        {text}""",
                        input_variables=["num_items", "text"],
                        partial_variables={
                            "format_instructions": parser.get_format_instructions(),
                        },
                    )

                    query = f"Genera un documento legal con un artículo para el siguiente fragmento de texto legal:\n{chunk}"

                    chain = LLMChain(llm=self._llm, prompt=prompt)
                    result = chain.invoke(query)

                    generated_doc_json = result['text']
                    print(
                        f"---\n\nThis is the generated doc json: {generated_doc_json}")
                    print(
                        f"This is the generated doc json type: {type(generated_doc_json)}\n")

                    generated_doc_dict = json.loads(generated_doc_json)

                    # Update the value of 'ley' to match the title of the legal document
                    generated_doc_dict['ley'] = title

                    print(
                        f"This is the generated doc dict: {generated_doc_dict}")
                    print(
                        f"This is the generated doc dict type: {type(generated_doc_dict)}\n")

                    # generated_doc = Article.model_validate_json(generated_doc_json)
                    # print(f"---\n\nThis is the generated doc: {generated_doc}")
                    # print(f"This is the generated doc type: {type(generated_doc)}\n")
                    articles.append(generated_doc_dict)

                    if len(articles) >= max_items_per_document:
                        break

                except Exception as e:
                    print(f"Failed to generate JSON objects for chunk: {e}")
                    continue

                time.sleep(1)

            legal_docs.append(articles)
            print(f"---\n\nThis is the legal docs: {legal_docs}")

        return legal_docs


def generate_instruction_output_pairs(
    self,
    json_objects: List[Dict],
    downstream_tasks: List[str],
    max_pairs_per_object: int,
    **kwargs,
) -> List[Dict]:
    """
    Generate instruction-output pairs for each JSON object based on the specified downstream tasks.

    :param json_objects: List of JSON objects representing articles.
    :param downstream_tasks: List of downstream tasks to generate pairs for.
    :param max_pairs_per_object: Maximum number of pairs to generate for each JSON object.
    :return: List of dictionaries containing instruction-output pairs.
    """
    instruction_output_pairs = []
    for json_object in json_objects:
        pairs = DownstreamTaskGenerator.generate_pairs(
            json_object,
            downstream_tasks,
            max_pairs_per_object,
            **kwargs,
        )
        instruction_output_pairs.extend(pairs)
    return instruction_output_pairs
