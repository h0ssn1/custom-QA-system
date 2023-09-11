# Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install farm-haystack[colab,faiss]
!pip install faiss-cpu

# Initialize the document store
from haystack.document_stores import FAISSDocumentStore
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

# Add data to the document store
from haystack.utils import clean_wiki_text, convert_files_to_docs
doc_dir = "/content/all_data"
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(docs)

# Initialize the retriever
from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
document_store.update_embeddings(retriever)

# Initialize the reader
from haystack.nodes import FARMReader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Run the pipeline
from haystack.pipelines import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# Get the answer
prediction = pipe.run(
    query="How can I study in Italy?",
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 10}}
)
print(prediction)
