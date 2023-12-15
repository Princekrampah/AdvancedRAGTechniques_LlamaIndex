from llama_index import (
    SimpleDirectoryReader,
    Document
)

from llama_index.node_parser import SentenceWindowNodeParser

# load document
documents = SimpleDirectoryReader(
    input_dir="../dataFiles/"
).load_data(show_progress=True)


# merge pages into one
document = Document(text="\n\n".join([doc.text for doc in documents]))

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
