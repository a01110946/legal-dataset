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
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm

from legal_document_loader import LegalDocumentLoader

class DatasetItem(BaseModel):
    question: str = Field(description="The question or instruction for the task.")
    answer: str = Field(description="The answer or output for the task.")
    context: str = Field(description="The context or input for the task. It very important to extract the text accurately and completely, following the format of the legal document. Please ensure that the content of the article is correct and well-structured. If the article is 'Article 22 bis', make sure to include the term 'bis' in the article number. If the content of the article contains bullet points or numbering, make sure to include it exactly as it appears in the legal document.")


class Dataset(BaseModel):
    items: List[DatasetItem]

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

        print(f"These are the legal documents: {legal_documents}\n\n")
        print(f"These are the legal documents type: {type(legal_documents)}\n---\n\n")

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

            articles = []

            for chunk in selected_chunks:
                try:                    
                    tasks = ", ".join(downstream_tasks)  
                    print(f"---\n\nThis are the tasks: {tasks}\n")
                    parser = JsonOutputParser(pydantic_object=DatasetItem)
                    prompt = PromptTemplate(
                        template="""
                        Tomando en cuenta el fragmento de texto legal compartido, por favor genera {max_pairs_per_article} ejemplos de pares instruction-output, 
                        para los siguientes tipos de tareas: "{tasks}".

                        El formato que deben seguir los ejemplos de tareas es el siguiente:\n\n{format_instructions}\n\n
                        
                        Para el valor de "context", debes asignar el texto extraído de los artículos que elegiste al generar "instruction" y "output",
                        es muy importante que extraigas el texto de manera precisa y completa, siguiendo el formato del documento legal.
                        Por favor, asegúrate de que el contenido del artículo sea correcto y esté bien estructurado. Si el artículo es 'Artículo 22 bis', asegúrate de incluir el término 'bis' en el número del artículo.
                        Si el contenido del artículo contiene viñetas o numeración, asegúrate de incluirlo también exactamente como se presenta en el documento.

                        A continuación, se presentan ejemplos de los tipos de tareas que se pueden generar a partir de este fragmento de texto legal:

                        Ejemplo de Question Answering (QA):
                        [
                        {{
                        "instruction": "De acuerdo con la Ley de Vertimientos en las Zonas Marinas Mexicanas, ¿qué información debe contener un permiso de vertimiento?",
                        "output": "Según el Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas, el permiso de vertimiento debe contener la siguiente información:\n\nI. Nombre de la persona física o denominación o razón social de la persona moral.\nII. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos.\nIII. Descripción de la materia o desecho a verter.\nIV. Denominación del Proyecto a desarrollarse.\nV. Vigencia del permiso.\nVI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana.\nVII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar por contravenir las disposiciones de la Ley o del permiso.\nVIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento.\nIX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento.",
                        "context": "El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."
                        }},
                        {{
                        "instruction": "Según la Ley General de Pesca y Acuacultura Sustentables, ¿cuál es el propósito del Sistema Nacional de Información de Pesca y Acuacultura?",
                        "output": "El propósito del Sistema Nacional de Información de Pesca y Acuacultura, según el Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables, es organizar, actualizar y difundir información sobre las actividades pesqueras y acuícolas que se desarrollan en el país.",
                        "context": "La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."
                        }}
                        ]

                        Ejemplos de Summarization:
                        [
                        {{
                        "instruction": "Resume el contenido del Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas.",
                        "output": "El Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas detalla los nueve requisitos específicos que debe contener un permiso de vertimiento, incluyendo información sobre el solicitante, el volumen y tipo de desechos, la ubicación del vertimiento, la garantía financiera para cubrir posibles daños, y los términos y condiciones establecidos por la Secretaría.",
                        "context": "El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."
                        }},
                        {{
                        "instruction": "Resume el contenido del Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables.",
                        "output": "El Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables establece que la Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura para organizar, actualizar y difundir información sobre actividades pesqueras y acuícolas. El sistema incluirá información de diversas fuentes, como la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, informes, tratados, acuerdos, resoluciones y estadísticas. Toda esta información deberá ser publicada en línea y en medios impresos.",
                        "context": "La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."
                        }}
                        ]

                        Ejemplos de Legal Advice Generation:
                        [
                        {{
                        "instruction": "Si una empresa desea obtener un permiso de vertimiento en zonas marinas mexicanas, ¿qué debe incluir en su solicitud según la Ley de Vertimientos en las Zonas Marinas Mexicanas?",
                        "output": "Para obtener un permiso de vertimiento en zonas marinas mexicanas, la empresa debe asegurarse de que su solicitud incluya toda la información requerida en el Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas. Esto incluye detalles sobre la empresa, el volumen y naturaleza de los desechos, la ubicación propuesta para el vertimiento, una garantía financiera para cubrir posibles daños, y aceptar los términos y condiciones establecidos por la Secretaría.",
                        "context": "El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."
                        }},
                        {{
                        "instruction": "Si un investigador quiere acceder a información sobre la pesca y acuacultura en México, ¿dónde puede encontrarla según la Ley General de Pesca y Acuacultura Sustentables?",
                        "output": "Según el Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables, el investigador puede encontrar información sobre la pesca y acuacultura en México en la página electrónica de la Secretaría y en los medios impresos que ésta ponga a disposición del público.",
                        "context": "La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."
                        }}
                        ]

                        Ejemplos de Legal Document Drafting:
                        [
                        {{
                        "instruction": "Redacta una cláusula para un contrato de vertimiento que detalle las responsabilidades del titular del permiso en caso de daños ambientales, de acuerdo con la Ley de Vertimientos en las Zonas Marinas Mexicanas.",
                        "output": "CLÁUSULA DÉCIMA QUINTA.- RESPONSABILIDAD AMBIENTAL. El Titular del Permiso se compromete a cumplir rigurosamente con todas las disposiciones de la Ley de Vertimientos en las Zonas Marinas Mexicanas y del permiso otorgado por la Secretaría. En caso de que el vertimiento cause daños al medio ambiente, el Titular del Permiso será responsable de la reparación de los mismos, así como de los perjuicios y multas que se pudieran aplicar, según lo establecido en el Artículo 22, fracción VII, de dicha Ley. El Titular del Permiso deberá contar con una garantía financiera suficiente para cubrir estos costos.",
                        "context": "El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."
                        }},
                        {{
                        "instruction": "Redacta una cláusula para un acuerdo de colaboración en materia de pesca y acuacultura que establezca el compromiso de compartir información relevante, de acuerdo con la Ley General de Pesca y Acuacultura Sustentables.",
                        "output": "CLÁUSULA OCTAVA.- INTERCAMBIO DE INFORMACIÓN. Las partes se comprometen a intercambiar información relevante sobre pesca y acuacultura, incluyendo datos estadísticos, resultados de investigaciones y cualquier otra información que pueda ser útil para el desarrollo sustentable de estas actividades. Dicho intercambio se realizará de conformidad con lo dispuesto en el Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables, respetando la confidencialidad de la información que así lo requiera.",
                        "context": "La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."
                        }}
                        ]

                        Por cada registro del JSON que te estoy compartiendo a continuación, genera un par instruction-output de cada tipo ("Question Answering (QA)", "Summarization", "Legal Advice Generation" y "Legal Document Drafting").
                        
                        Context: {chunk}
                        """,
                        input_variables=["max_pairs_per_article", "context", "tasks", "chunk"],
                        partial_variables={
                            "format_instructions": parser.get_format_instructions(),
                        },
                    )

                    query = f"Genera {max_pairs_per_article} ejemplos de pares instruction-output, para las siguientes tareas {', '.join(downstream_tasks)}, con el siguiente fragmento de texto legal:\n{chunk}"
                    
                    chain = LLMChain(llm=self._llm, prompt=prompt, max_pairs_per_article=max_pairs_per_article, context=chunk, tasks=downstream_tasks)
                    result = chain.invoke(query)

                    generated_doc_json = result['text']
                    print(f"---\n\nThis is the generated doc json: {generated_doc_json}")
                    print(f"This is the generated doc json type: {type(generated_doc_json)}\n")

                    generated_doc_dict = json.loads(generated_doc_json)

                    # Update the value of 'ley' to match the title of the legal document
                    #generated_doc_dict['ley'] = title

                    print(f"This is the generated doc dict: {generated_doc_dict}")
                    print(f"This is the generated doc dict type: {type(generated_doc_dict)}\n")

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
    