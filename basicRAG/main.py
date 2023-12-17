from typing import List
from llama_index import (
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    load_index_from_storage
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.query_engine import RetrieverQueryEngine

from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as OpenAITruLens

import numpy as np

# for loading environment variables
from decouple import config

import os

# set env variables
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


# create LLM and Embedding Model
embed_model = OpenAIEmbedding()
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm=llm)


# check if data indexes already exists
if not os.path.exists("./storage"):
    # load data
    documents = SimpleDirectoryReader(
        input_dir="../dataFiles").load_data(show_progress=True)

    # create nodes parser
    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

    # split into nodes
    base_nodes = node_parser.get_nodes_from_documents(documents=documents)

    # creating index
    index = VectorStoreIndex(nodes=base_nodes, service_context=service_context)

    # store index
    index.storage_context.persist()
else:
    # load existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context=storage_context)


# create retriever
retriever = index.as_retriever(similarity_top_k=2)

# query retriever engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    service_context=service_context
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


tru_query_engine_recorder = TruLlama(query_engine,
    app_id='Basic_RAG',
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
            query_engine.query(qn)


run_eval(eval_questions=eval_questions)

# run dashboard
tru.run_dashboard()
