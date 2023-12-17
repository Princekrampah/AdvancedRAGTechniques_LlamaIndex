import os
from typing import List

from llama_index import (
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage
)

from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank


from llama_index.llms import OpenAI

# for loading environment variables
from decouple import config

from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as OpenAITruLens

import numpy as np

# set env variables
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


# load document
documents = SimpleDirectoryReader(
    input_dir="../dataFiles/"
).load_data(show_progress=True)


# merge pages into one
document = Document(text="\n\n".join([doc.text for doc in documents]))


def create_indexes(
    documents: Document,
    index_save_dir: str,
    window_size: int = 4,
    llm_model: str = "gpt-3.5-turbo",
    temperature: float = 0.1
):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # creating OpenAI gpt-3.5-turbo LLM and OpenAIEmbedding model
    llm = OpenAI(model=llm_model, temperature=temperature)
    embed_model = OpenAIEmbedding()

    # creating the service context
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    if not os.path.exists(index_save_dir):
        # creating the vector store index
        index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )

        # make vector store persistant
        index.storage_context.persist(persist_dir=index_save_dir)
    else:
        # load vector store indexed if they exist
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_save_dir),
            service_context=sentence_context
        )

    return index


def create_query_engine(
    sentence_index: VectorStoreIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 5,
    rerank_model: str = "BAAI/bge-reranker-base",
):
    # add meta data replacement post processor
    postproc = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    # link: https://huggingface.co/BAAI/bge-reranker-base
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model=rerank_model
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank]
    )

    return sentence_window_engine


# create index
index = create_indexes(
    documents=documents,
    index_save_dir="./sentence_window_size_10_index",
    window_size=3,
    llm_model="gpt-3.5-turbo",
    temperature=0.1
)

# create query engine
sentence_window_engine = create_query_engine(
    sentence_index=index,
    similarity_top_k=5,
    rerank_top_n=2,
)

# RAG pipeline evals
tru = Tru(database_file="../default.sqlite")

openai = OpenAITruLens()

grounded = Groundedness(groundedness_provider=OpenAITruLens())

# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text
).on_output(
).aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.mean)


tru_query_engine_recorder = TruLlama(sentence_window_engine,
                                     app_id='sentence_window_size_10',
                                     feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])


eval_questions = []

with open("./eval_questions.txt", "r") as eval_qn:
    for qn in eval_qn:
        qn_stripped = qn.strip()
        eval_questions.append(qn_stripped)


def run_eval(eval_questions: List[str]):
    for qn in eval_questions:
        # eval using context window
        with tru_query_engine_recorder as recording:
            sentence_window_engine.query(qn)


run_eval(eval_questions=eval_questions)

# run dashboard
tru.run_dashboard()
