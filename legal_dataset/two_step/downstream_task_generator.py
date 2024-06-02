# downstream_task_generator.py
"""
Module: downstream_task_generator.py
Description: Module for generating instruction-output pairs for downstream tasks.
"""

import json
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain

"""
class QuestionAnsweringInstruction(BaseModel):
    instruction: str = Field(
        description="The instruction or question for the Question Answering task.")
    output: str = Field(
        description="The corresponding answer or output for the Question Answering instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to generate the Question Answering instruction and output.")

class SummarizationInstruction(BaseModel):
    instruction: str = Field(
        description="The instruction or prompt for the Summarization task.")
    output: str = Field(
        description="The corresponding summary or output for the Summarization instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to generate the Summarization instruction and output.")

class LegalAdviceInstruction(BaseModel):
    instruction: str = Field(
        description="The instruction or prompt for the Legal Advice task.")
    output: str = Field(
        description="The corresponding advice or output for the Legal Advice instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to generate the Legal Advice instruction and output.")

class DocumentDraftingInstruction(BaseModel):
    instruction: str = Field(
        description="The instruction or prompt for the Document Drafting task.")
    output: str = Field(
        description="The corresponding drafted document or output for the Document Drafting instruction.")
    context: str = Field(
        description="The relevant excerpt from the legal document used to generate the Document Drafting instruction and output.")
"""

class Tasks(BaseModel):
    question_answering: Dict = Field(
        ...,
        description="Se refiere a una lista de tareas de Pregunta-Respuesta que siguen el formato 'instruction-output-context', donde 'instruction' se refiere a la instrucción que se otorga a un LLM; 'output' se refiere a la respuesta que entrega el LLM; y 'context' se refiere al texto proporcionado como contexto para realizar la instrucción.",
        examples=['{\
        "instruction": "De acuerdo con la Ley de Vertimientos en las Zonas Marinas Mexicanas, ¿qué información debe contener un permiso de vertimiento?",\
        "output": "Según el Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas, el permiso de vertimiento debe contener la siguiente información:\n\nI. Nombre de la persona física o denominación o razón social de la persona moral.\nII. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos.\nIII. Descripción de la materia o desecho a verter.\nIV. Denominación del Proyecto a desarrollarse.\nV. Vigencia del permiso.\nVI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana.\nVII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar por contravenir las disposiciones de la Ley o del permiso.\nVIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento.\nIX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento.",\
        "context": "Ley de Vertimientos en las Zonas Marinas Mexicanas - Artículo 22. El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."\
        }',
        '{\
        "instruction": "Según la Ley General de Pesca y Acuacultura Sustentables, ¿cuál es el propósito del Sistema Nacional de Información de Pesca y Acuacultura?",\
        "output": "El propósito del Sistema Nacional de Información de Pesca y Acuacultura, según el Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables, es organizar, actualizar y difundir información sobre las actividades pesqueras y acuícolas que se desarrollan en el país.",\
        "context": "Ley General de Pesca y Acuacultura Sustentables - Artículo 120. La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."\
        }'])
    summarization: Dict = Field(
        ...,
        description="Se refiere a una lista de tareas de Resumen que siguen el formato 'instruction-output-context', donde 'instruction' se refiere a la instrucción que se otorga a un LLM; 'output' se refiere a la respuesta que entrega el LLM; y 'context' se refiere al texto proporcionado como contexto para realizar la instrucción.",
        examples=['{\
        "instruction": "Resume el contenido del Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas.",\
        "output": "El Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas detalla los nueve requisitos específicos que debe contener un permiso de vertimiento, incluyendo información sobre el solicitante, el volumen y tipo de desechos, la ubicación del vertimiento, la garantía financiera para cubrir posibles daños, y los términos y condiciones establecidos por la Secretaría.",\
        "context": "Ley de Vertimientos en las Zonas Marinas Mexicanas - Artículo 22. El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."\
        },\
        {\
        "instruction": "Resume el contenido del Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables.",\
        "output": "El Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables establece que la Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura para organizar, actualizar y difundir información sobre actividades pesqueras y acuícolas. El sistema incluirá información de diversas fuentes, como la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, informes, tratados, acuerdos, resoluciones y estadísticas. Toda esta información deberá ser publicada en línea y en medios impresos.",\
        "context": "Ley General de Pesca y Acuacultura Sustentables - Artículo 120. La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."\
        }'])
    legal_advice: Dict = Field(
        ...,
        description="Se refiere a una lista de tareas de Aconsejamiento Legal que siguen el formato 'instruction-output-context', donde 'instruction' se refiere a la instrucción que se otorga a un LLM; 'output' se refiere a la respuesta que entrega el LLM; y 'context' se refiere al texto proporcionado como contexto para realizar la instrucción.",
        examples=['{\
        "instruction": "Si una empresa desea obtener un permiso de vertimiento en zonas marinas mexicanas, ¿qué debe incluir en su solicitud según la Ley de Vertimientos en las Zonas Marinas Mexicanas?",\
        "output": "Para obtener un permiso de vertimiento en zonas marinas mexicanas, la empresa debe asegurarse de que su solicitud incluya toda la información requerida en el Artículo 22 de la Ley de Vertimientos en las Zonas Marinas Mexicanas. Esto incluye detalles sobre la empresa, el volumen y naturaleza de los desechos, la ubicación propuesta para el vertimiento, una garantía financiera para cubrir posibles daños, y aceptar los términos y condiciones establecidos por la Secretaría.",\
        "context": "Ley de Vertimientos en las Zonas Marinas Mexicanas - Artículo 22. El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."\
        },\
        {\
        "instruction": "Si un investigador quiere acceder a información sobre la pesca y acuacultura en México, ¿dónde puede encontrarla según la Ley General de Pesca y Acuacultura Sustentables?",\
        "output": "Según el Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables, el investigador puede encontrar información sobre la pesca y acuacultura en México en la página electrónica de la Secretaría y en los medios impresos que ésta ponga a disposición del público.",\
        "context": "Ley General de Pesca y Acuacultura Sustentables - Artículo 120. La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."\
        }'])
    document_drafting: Dict = Field(
        ...,
        description="Se refiere a una lista de tareas de Redacción de Documentos Legales que siguen el formato 'instruction-output-context', donde 'instruction' se refiere a la instrucción que se otorga a un LLM; 'output' se refiere a la respuesta que entrega el LLM; y 'context' se refiere al texto proporcionado como contexto para realizar la instrucción.",
        examples=['{\
        "instruction": "Redacta una cláusula para un contrato de vertimiento que detalle las responsabilidades del titular del permiso en caso de daños ambientales, de acuerdo con la Ley de Vertimientos en las Zonas Marinas Mexicanas.",\
        "output": "CLÁUSULA DÉCIMA QUINTA.- RESPONSABILIDAD AMBIENTAL. El Titular del Permiso se compromete a cumplir rigurosamente con todas las disposiciones de la Ley de Vertimientos en las Zonas Marinas Mexicanas y del permiso otorgado por la Secretaría. En caso de que el vertimiento cause daños al medio ambiente, el Titular del Permiso será responsable de la reparación de los mismos, así como de los perjuicios y multas que se pudieran aplicar, según lo establecido en el Artículo 22, fracción VII, de dicha Ley. El Titular del Permiso deberá contar con una garantía financiera suficiente para cubrir estos costos.",\
        "context": "Ley de Vertimientos en las Zonas Marinas Mexicanas - Artículo 22. El permiso de vertimiento contendrá: I. Nombre de la persona física o la denominación o razón social de la persona moral, según corresponda; II. Volumen de los desechos u otras materias a verter expresadas en metros cúbicos; III. Descripción de la materia o desecho a verter; IV. Denominación del Proyecto a desarrollarse; V. Vigencia del permiso; VI. Situación geográfica y profundidad de la zona de tiro autorizada, así como la distancia a la costa más cercana; VII. La cantidad que garantice la reparación de los daños, perjuicios y multa que se pudieran aplicar, por contravenir las disposiciones de la Ley o del permiso; VIII. Los términos y condicionantes que establezca la Secretaría para llevar a cabo el vertimiento, y IX. La obligación del titular del permiso de responder por los daños al medio ambiente que pudiera ocasionar el vertimiento."\
        },\
        {\
        "instruction": "Redacta una cláusula para un acuerdo de colaboración en materia de pesca y acuacultura que establezca el compromiso de compartir información relevante, de acuerdo con la Ley General de Pesca y Acuacultura Sustentables.",\
        "output": "CLÁUSULA OCTAVA.- INTERCAMBIO DE INFORMACIÓN. Las partes se comprometen a intercambiar información relevante sobre pesca y acuacultura, incluyendo datos estadísticos, resultados de investigaciones y cualquier otra información que pueda ser útil para el desarrollo sustentable de estas actividades. Dicho intercambio se realizará de conformidad con lo dispuesto en el Artículo 120 de la Ley General de Pesca y Acuacultura Sustentables, respetando la confidencialidad de la información que así lo requiera.",\
        "context": "Ley General de Pesca y Acuacultura Sustentables - Artículo 120. La Secretaría integrará el Sistema Nacional de Información de Pesca y Acuacultura que tendrá por objeto organizar, actualizar y difundir la información sobre actividades pesqueras y acuícolas, particularmente las que se desarrollan en el país. El sistema se integrará con la información de la Carta Nacional Pesquera, la Carta Nacional Acuícola, el Registro Nacional de Pesca y Acuacultura, el Informe de la situación general de la pesca y acuacultura en México e indicadores de su desarrollo, los Tratados y Acuerdos Internacionales en las materias mencionadas, las resoluciones definitivas acerca de concesiones, permisos y para realizar actividades pesqueras y acuícolas, y el anuario estadístico de pesca y acuacultura. Toda esta información deberá ser publicada en la página electrónica de la Secretaría y por los medios impresos a su alcance."\
        }'])
    legal_document: str = Field(
        description="Se refiere al título del documento al que pertenece toda la información de Tasks."
    )
    article: str = Field(
        description="Se refiere al número de artículo para el cual estamos generando la tarea."
    )

class Dataset(BaseModel):
    items: List[Tasks]

class DownstreamTaskGenerator:
    """
    Class for generating instruction-output pairs for downstream tasks.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the DownstreamTaskGenerator.

        :param llm: The language model to use for generating the instruction-output pairs for downstream tasks.
        """
        self._llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)

    def generate_pairs(
        self,
        legal_documents: List[List[Dict]],
        #downstream_tasks: List[str],
        #max_pairs_per_article: int,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate instruction-output pairs for each article in the legal documents based on the specified downstream tasks.

        :param legal_documents: List of lists of dictionaries, where each inner list represents a legal document and each dictionary represents an article.
        :param downstream_tasks: List of downstream tasks to generate pairs for.
        :param max_pairs_per_article: Maximum number of pairs to generate for each article.
        :return: List of dictionaries containing instruction-output pairs.
        """
        instruction_output_pairs = []

        parser = JsonOutputParser(pydantic_object=Tasks)
        prompt = PromptTemplate(
            template = """
            Por cada registro del JSON que te estoy compartiendo a continuación, genera un par instruction-output de cada tipo ("Question Answering (QA)", "Summarization", "Legal Advice Generation" y "Legal Document Drafting").
            
            Recuerda que el formato del artículo debe seguir el siguiente esquema:
            :\n\n
            {format_instructions}\n\n

            Contexto: {context}
            """,
            input_variables=["context"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )

        for legal_document in legal_documents:
            for article in legal_document:
                context = (f"{article['ley']} - {article['articulo_no']}. {article['articulo_contenido']}")
                query = f"Genera un par instruction-output de cada tipo de tarea el siguiente artículo:\n{context}" 
                chain = LLMChain(llm=self._llm, prompt=prompt)
                result = chain.invoke(query)
                print(f"These are the results: {result}")

                pairs = self._parse_ai_message(result, article['ley'], article['articulo_no'])
                print(f"These are the pairs: {pairs}")
                instruction_output_pairs.extend(pairs)

        print(f"Generated a total of {len(instruction_output_pairs)} instruction-output pairs.")
        print(f"The type for instruction-output pairs is: {type(instruction_output_pairs)}")        
        print(f"These are the instruction-output pairs: {instruction_output_pairs}")
        print(f"The nested type for instruction-output pairs is: {type(instruction_output_pairs[0])}")
        print(f"These are the instruction-output pairs: {instruction_output_pairs}")
        
        return Dataset(items=instruction_output_pairs)

    def _parse_ai_message(self, content: dict, legal_document: str = None, article: str = None) -> List[Dict]:
        pairs = []

        if "text" in content:
            try:
                json_string = content["text"]
                json_string = json_string.replace("\n", "")
                json_data = json.loads(json_string)
                if legal_document is not None:
                    json_data['legal_document'] = legal_document
                if article is not None:
                    json_data['article'] = article
                print(f"This is json_data: {json_data}")
                print(f"This is the type of json_data: {type(json_data)}")
                print(f"This are the keys of json_data: {json_data.keys()}")
                tasks = Tasks(**json_data)

                pairs.append(tasks)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Content: {content['text']}")
            except KeyError as e:
                print(f"Missing key in JSON data: {e}")
                print(f"Content: {content['text']}")

        return pairs