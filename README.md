# Coursera Course: LangChain Chat with Your Data
Link: https://www.coursera.org/learn/langchain-chat-with-your-data-project/home/week/1

## Project Ideas
- Question Answering over inner documentation data for a department of a company

## Introduction
LangChain is an Open-source developer framework for building LLM applications. Its made up of multiple components:
- Prompts.
- Chains.
- Models.
- Indexes.
- Agents.

## LangChain steps
1. Document Loading
2. Document splitting
3. Storage
4. Retrieval
5. Output via QA


### Document Loading
Loaders deal with specific sources for converting the data as: PDF, Youtube, HTML, JSON, Word...etc. The main objective is load them in a unified format. The data can be unstructured and structured.

```
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader

# For example, in order to load any data use a specific loader
loader = Loader(DATA_URL_LINK_PATH)
data = loader.load()
```

### Document Splitting
Document Splitting is required to create chunks from the data in order to work with batches of data. This splitting can be problematic by having a same sentence splitted in different chunks. This is required to obtain semantically relevant chunks. All the splitters work with a chunk size and some chunk overlap. 

```
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]  # Default separators
)  # Splits recursively by the order of the separator oposed to the CharacterTextSplitter

r_splitter.split_text(text)

```

Another way to split is with the `TokenTextSplitter`. It splits by tokens:

```
from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

```

With the `MarkdownHeaderTextSplitter`, its possible to split data as github Markdown:

```
from langchain.text_splitter import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
```

### Vector Stores and Embeddings
The chunks obtained are managed by indexes in order to answer to questions about the data. Embedding are obtained from the splitted text and stored with Vector Stores. Sentences which are similar have similar embeddings. An usage example:
1. Splitted text is converted to embeddings
2. Question is also converted to an embedding
3. Comparison of the embeddings is done to obtain the n most similar
4. Lastly, the LLM processes the data in order to obtain an appropiate result.

Extract the embeddings with the `OpenAIEmbeddings`:
```
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

embedding1 = embedding.embed_query(sentence1)
```

The Vectorstores can be stored on Chroma and then similarity is done with:
```
from langchain.vectorstores import Chroma
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

docs = vectordb.similarity_search(question,k=3)  # Obtain K most similar samples.

vectordb.persist() # Save for future usage
```

### Retrieval
The technique `Maximum marginal relevance` helps to obtain diverse results. The algorithm works by a first `fetch_k` documents and from this set choose the `k` most diverse.
```
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
```
In some questions, it is of interest of the user to obtain data by searching with a term, but also with some specific metadata (e.g.: specific date, location...etc). In this case, we need to parse from the question the required data for the query and this can be done with a LLM. This technique is called selfquery:
```
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name=attribute_name,
        description=description,
        type=string,
    ),
    ...etc.
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
docs = retriever.get_relevant_documents(question)
```
Finally, with compression the data is passed to a LLM which filters out irrevelant data. 
```
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
compressed_docs = compression_retriever.get_relevant_documents(question)
```

### Question Answering
Next step of the sequence is get the question and the documents passed to a LLM and obtain the response. When the documents are to large to fit into the LLM context it is useful to follow the next methods:
- Map reduce. It uses the LLM to process an answer for each of the chunks before processing one complete answer
- Refine. Add step by step the data via the LLM in order to refine the answer.
- Map rerank. Make the LLM score the answer chunks and select the chunk of highest score.

```
from langchain.chains import RetrievalQA

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type="map_reduce",  # OR refine or map_rerank
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
```

### Chat
It is useful to maintain memory along the session for having questions which are related to previous one. This function can be managed by a chatbot. 

```
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
result = qa({"question": question})
print(result['answer'])
```

## Tips and tricks
- Force variety of search results with `Maximum marginal relevance`.
- Compress relevant splits to fit into LLM context.
- Send information of question and response along to the LLM to obtain the answer.
- Langchain tracing for view the retrieval process (langchain plus platform).
