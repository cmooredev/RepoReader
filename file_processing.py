# file_processing.py
import os
import uuid
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain.document_loaders import DirectoryLoader, NotebookLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import clean_and_tokenize
import unicodedata
import traceback

def extract_repo_name(repo_url):
    # Extract the part of the URL after the last slash and before .git
    repo_name = repo_url.split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]  # remove .git from the end
    return repo_name


def is_repo_cloned(repo_url, path_dir):
    repo_name = extract_repo_name(repo_url)
    repo_path = os.path.join(path_dir, repo_name)
    return os.path.isdir(repo_path)

def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False

def load_and_index_files(repo_path):
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala',
                  'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css',
                  'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

    file_type_counts = {}
    documents_dict = {}

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext == 'ipynb':
                loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20,
                                        remove_newline=True)
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern, loader_kwargs={"content_type": "text/plain"})

            loaded_documents = loader.load() if callable(loader.load) else []
            print(f'[LOG] {ext} loaded!')
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id

                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            # print(traceback.format_exc())
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    split_documents = []
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']

        split_documents.extend(split_docs)

    index = None
    if split_documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
        index = BM25Okapi(tokenized_documents)
    return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]

def search_documents(query, index, documents, n_results=5):
    query_tokens = clean_and_tokenize(query)
    bm25_scores = index.get_scores(query_tokens)

    # Compute TF-IDF scores
    try:
        tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True, encoding="utf-8")
        # Normalize and clean up the page content, handling any special characters
        documents_cleaned = [unicodedata.normalize('NFKD', doc.page_content).encode('ascii', 'ignore').decode('utf-8') for
                             doc in documents]

        tfidf_matrix = tfidf_vectorizer.fit_transform(documents_cleaned)
        query_tfidf = tfidf_vectorizer.transform([query])

        # Compute Cosine Similarity scores
        cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # Combine BM25 and Cosine Similarity scores
        combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5

        # Get unique top documents
        unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]

    except Exception as e:
        print('[ERROR]: TfidfVectorizer')
        print('[ERROR]: documents: {documents_cleaned}')
        raise e



    return [documents[i] for i in unique_top_document_indices]