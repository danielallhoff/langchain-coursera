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
