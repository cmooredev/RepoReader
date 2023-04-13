# file_processing.py
import os
import uuid
import subprocess
from rank_bm25 import BM25Okapi
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import clean_and_tokenize

def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False

def load_and_index_files(repo_path):
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

    file_type_counts = {}
    documents_dict = {}
    documents = []
    filenames = []

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext == 'ipynb':
                loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = loader.load() if callable(loader.load) else []
            documents += loaded_documents

            if documents:
                file_type_counts[ext] = len(documents)
            for doc in documents:
                file_path = doc.metadata['source']
                relative_path = os.path.relpath(file_path, repo_path)
                file_id = str(uuid.uuid4())
                doc.metadata['source'] = relative_path
                doc.metadata['file_id'] = file_id
                filenames.append(relative_path)

                documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    split_documents_dict = {}
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']

        split_documents_dict[file_id] = split_docs

    documents = [split_doc for split_docs in split_documents_dict.values() for split_doc in split_docs]

    index = None
    if documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in documents]
        index = BM25Okapi(tokenized_documents)
    return index, documents, file_type_counts, filenames
