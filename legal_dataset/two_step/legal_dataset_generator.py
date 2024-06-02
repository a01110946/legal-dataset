# legal_dataset_generator.py

"""
Module: legal_dataset_generator.py
Description: Module for generating legal datasets from Mexican legal documents.
"""

import json
import random
import time
from typing import Dict, List

from downstream_task_generator import DownstreamTaskGenerator
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from legal_document_loader import LegalDocumentLoader
from llm.openai import chat_completion_request
from pydantic import BaseModel, Field
from tqdm import tqdm
from utf8_encoder import convert_to_utf8
from spanish_title_case import spanish_title_case


class Article(BaseModel):
    ley: str = Field(
        description="Se refiere al nombre de la ley.",
        examples=["Código Federal de Procedimientos Civiles", "Ley General de Cambio Climático"])
    titulo_no: str = Field(
        description="Se refiere al número de título. Siempre debe encontrarse antes del Capítulo; si solo encuentras un título después del capítulo, entonces declara 'N/A'.",
        examples=["Título Segundo", "Título Tercero"])
    titulo_nombre: str = Field(
        description="Se refiere al nombre del título. Siempre debe encontrarse antes del Capítulo; si solo encuentras un título después del capítulo, entonces declara 'N/A'.",
        examples=["De la Sucesión por Testamento"])
    capitulo_no: str = Field(
        description="Se refiere al número de capítulo. Siempre debe encontrarse antes del Artículo; si solo encuentras un capítulo después del artículo, entonces declara 'N/A'.",
        examples=["Capítulo I", "Capítulo V", "Capítulo XII", "Capítulo Único"])
    capitulo_nombre: str = Field(
        description="Se refiere al nombre del capítulo. Siempre debe encontrarse antes del Artículo; si solo encuentras un capítulo después del artículo, entonces declara 'N/A'.",
        examples=["De los Testamentos en General", "Disposiciones Generales", "De la Planeación, Programación y Prespuesto"])
    articulo_no: str = Field(
        description="Se refiere al número de artículo.",
        examples=["Artículo 1413", "Artículo 36 Bis"])
    articulo_contenido: str = Field(
        description="Se refiere al contenido del artículo. Siempre debe incluir el texto completo del artículo, incluyendo viñetas y numeraciones.",
        examples=["Queda también sin efecto el legado, si el testador enajena la cosa legada; pero vale si la recobra por un título legal.",
                  "Para los efectos de esta Ley, se entenderá por: \
                    \
                    I. Autoproducción de vivienda: el proceso de gestión de suelo, construcción y distribución de vivienda \
                    bajo el control directo de sus usuarios de forma individual o colectiva, la cual puede desarrollarse \
                    mediante la contratación de terceros o por medio de procesos de autoconstrucción; \
                    \
                    \
                    II. Autoconstrucción de vivienda: el proceso de construcción o edificación de la vivienda realizada \
                    directamente por sus propios usuarios, en forma individual, familiar o colectiva; \
                    \
                    III. Estímulos: las medidas de carácter jurídico, administrativo, fiscal o financiero que establezcan los \
                    diferentes órdenes de gobierno para promover y facilitar la participación de los sectores social y privado, \
                    en la ejecución de acciones, procesos o programas habitacionales; \
                    \
                    IV. Espacios Habitables: el lugar de la vivienda donde se desarrollan actividades de reunión o \
                    descanso, que cuenten con las dimensiones mínimas de superficie, altura, ventilación e iluminación \
                    natural, además de contar como mínimo con un baño, cocina, estancia-comedor y dos recamaras, de \
                    conformidad con las características y condiciones mínimas necesarias que establezcan las leyes y las \
                    normas oficiales mexicanas; \
                    \
                    V. Espacios Auxiliares: el lugar de la vivienda donde se desarrollan actividades de trabajo, higiene y \
                    circulación; \
                    \
                    VI. Comisión: la Comisión Nacional de Vivienda; \
                    \
                    VII. Comisión Intersecretarial: la Comisión Intersecretarial de Vivienda; \
                    \
                    VIII. Consejo: el Consejo Nacional de Vivienda; \
                    \
                    IX. Mejoramiento de vivienda: la acción tendiente a consolidar o renovar las viviendas deterioradas \
                    física o funcionalmente, mediante actividades de ampliación, reparación, reforzamiento estructural o \
                    rehabilitación que propicien una vivienda digna y decorosa; \
                    \
                    X. Producción social de vivienda: aquella que se realiza bajo el control de autoproductores y \
                    autoconstructores que operan sin fines de lucro y que se orienta prioritariamente a atender las \
                    necesidades habitacionales de la población de bajos ingresos, incluye aquella que se realiza por \
                    procedimientos autogestivos y solidarios que dan prioridad al valor de uso de la vivienda por sobre la \
                    definición mercantil, mezclando recursos, procedimientos constructivos y tecnologías con base en sus \
                    propias necesidades y su capacidad de gestión y toma de decisiones; \
                    \
                    XI. Productor social de vivienda: la persona física o moral que en forma individual o colectiva produce \
                    vivienda sin fines de lucro;\
                    \
                    XII. Política Nacional de Vivienda: el conjunto de disposiciones, criterios, lineamientos y medidas de \
                    carácter general que se establecen para coordinar las acciones de vivienda que realicen las autoridades \
                    federales, de las entidades federativas y municipales, así como su concertación con los sectores privado \
                    y social, con la finalidad de cumplir con el mandato constitucional del derecho a la vivienda digna y \
                    decorosa; \
                    \
                    XIII. Sistema de Información: el Sistema Nacional de Información e Indicadores de Vivienda, como el \
                    conjunto de datos producidos por los sectores público, social y privado, organizados bajo una estructura \
                    conceptual predeterminada, que permita mostrar la situación de la vivienda y el mercado habitacional, así \
                    como los efectos de las políticas públicas en la materia, y \
                    \
                    XIV. Suelo: los terrenos física y legalmente susceptibles de ser destinados predominantemente al uso \
                    habitacional conforme a las disposiciones aplicables."])


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

        # print(f"These are the legal documents: {legal_documents}\n\n")
        # print(
            # f"These are the legal documents type: {type(legal_documents)}\n---\n\n")

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

        #print(f"---\n\nThis is the legal documents: {legal_documents}\n")
        #print(f"This is the legal documents type: {type(legal_documents)}\n")
        #print(
        #    f"This is the type of the first legal document: {type(legal_documents[0])}\n")
        # print(
        #    f"These are the number of legal documents: {len(legal_documents)}\n")
        # print(
        #    f"These are the keys of the legal documents: {legal_documents[0].keys()}\n")

        for document in legal_documents:
            #print(f"---\n\nThis is the doc: {document}\n")
            #print(f"This is the doc type: {type(document)}\n")
            #print(
            #    f"This is a sample from the doc text: {document['Text'][700:900]}\n")

            #title = convert_to_utf8(document["Title"].encode('latin-1'))
            title = spanish_title_case(document["Title"])
            print(f"This is the doc title: {title}\n")
            
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

            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\nTITULO ", "\nTítulo", "\n\n", "\n"], keep_separator=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
            text_chunks = text_splitter.split_text(text)
            print(f'This is the length of the text chunk: {len(text_chunks)}')
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
                        {chunk}""",
                        input_variables=["num_items", "chunk"],
                        partial_variables={
                            "format_instructions": parser.get_format_instructions(),
                        },
                    )

                    query = f"Genera un documento legal con un artículo para el siguiente fragmento de texto legal:\n{chunk}"
                    print(query)

                    chain = LLMChain(llm=self._llm, prompt=prompt)
                    result = chain.invoke(query)

                    generated_doc_json = result['text']
                    #print(
                    #    f"---\n\nThis is the generated doc json: {generated_doc_json}")
                    #print(
                    #    f"This is the generated doc json type: {type(generated_doc_json)}\n")

                    generated_doc_dict = json.loads(generated_doc_json)

                    # Update the value of 'ley' to match the title of the legal document
                    generated_doc_dict['ley'] = title

                    print(
                        f"This is the generated doc dict: {generated_doc_dict}")
                    #print(
                    #    f"This is the generated doc dict type: {type(generated_doc_dict)}\n")

                    # generated_doc = Article.model_validate_json(generated_doc_json)
                    # print(f"---\n\nThis is the generated doc: {generated_doc}")
                    # print(f"This is the generated doc type: {type(generated_doc)}\n")
                    articles.append(generated_doc_dict)

                    time.sleep(21)

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
