import os
import uuid
from pathlib import Path
import subprocess
import tempfile
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import re
import nltk

nltk.download("punkt")

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"

model_name = "gpt-3.5-turbo"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def clean_and_tokenize(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(?:http|ftp)s?://\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return nltk.word_tokenize(text)

def format_documents(documents):
    numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content}" for i, doc in enumerate(documents)])
    return numbered_docs

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
                loader = NotebookLoader(str(notebook_path), include_outputs=True, max_output_length=20, remove_newline=True)
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = loader.load() if callable(loader.load) else []
            documents += loaded_documents

            if documents:
                file_type_counts[ext] = len(documents)
            for doc in documents:
                file_path = doc.metadata['source']
                relative_path = os.path.relpath(file_path, repo_path)
                file_id = str(uuid.uuid4())  # Generate a unique file identifier as a string
                doc.metadata['source'] = relative_path
                doc.metadata['file_id'] = file_id  # Add the file identifier to the metadata
                filenames.append(relative_path) 

                # Add the document to the dictionary using the file_id as the key
                documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    split_documents_dict = {}
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            file_id = split_doc.metadata['file_id']
            original_doc = documents_dict[file_id]
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']

        split_documents_dict[file_id] = split_docs

    documents = [split_doc for split_docs in split_documents_dict.values() for split_doc in split_docs]

    for i, (split_doc, original_doc) in enumerate(zip(documents, documents_dict.values())):
        split_doc.metadata['file_id'] = original_doc.metadata['file_id']

    embeddings = OpenAIEmbeddings()
    index = None
    if documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in documents]
        index = BM25Okapi(tokenized_documents)
    return index, documents, file_type_counts, filenames

def ask_question(documents, question, llm_chain, repo_name, github_url, conversation_history, file_type_counts, filenames):
    tokenized_question = clean_and_tokenize(question)
    scores = index.get_scores(tokenized_question)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    similar_docs = [documents[i] for i in top_k_indices]

    relevant_docs = [doc for doc in similar_docs if len(set(tokenized_question).intersection(clean_and_tokenize(doc.page_content))) >= 3]

    numbered_documents = format_documents(relevant_docs)
    context = f"This question is about the GitHub repository '{repo_name}' available at {github_url}. The most relevant documents are:\n\n{numbered_documents}"

    answer_with_sources = llm_chain.run(
        model=model_name,
        question=question,
        context=context,
        repo_name=repo_name,
        github_url=github_url,
        conversation_history=conversation_history,
        numbered_documents=numbered_documents
    )
    return answer_with_sources

def format_user_question(question):
    question = re.sub(r'\s+', ' ', question).strip()
    return question

if __name__ == "__main__":
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    repo_link = github_url
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                print("No documents were found to index. Exiting.")
                exit()

            print("Repository cloned. Indexing files...")
            llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
            
            template = """
            Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question}

            Instr:
            1. Answer based on context/docs.
            2. Focus on repo/code.
            3. Consider:
            a. Purpose/features - describe.
            b. Functions/code - provide details/samples.
            c. Setup/usage - give instructions.
            d. List files - share relevant names.
            4. Unsure? Say "I am not sure".

            Answer: 
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            conversation_history = ""
            while True:
                try:
                    user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == "exit()":
                        break

                    user_question = format_user_question(user_question)

                    answer = ask_question(documents, user_question, llm_chain, repo_name, github_url, conversation_history, file_type_counts, filenames)
                    print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                    conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
            print("Failed to clone the repository.")
