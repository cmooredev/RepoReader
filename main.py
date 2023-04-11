import os
import uuid
from pathlib import Path
import subprocess
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def format_documents(documents):
    numbered_docs = "\n".join([f"{i+1}. {doc.metadata['source']}: {doc.page_content}" for i, doc in enumerate(documents)])
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
    filenames = []
    # Use a dictionary to store documents with their respective file_ids as keys
    documents_dict = {}
    documents = []
    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            if ext == 'ipynb':
                # Load .ipynb files using NotebookLoader
                print(f"Loading .ipynb files with pattern '{glob_pattern}'")
                notebook_documents = []
                for notebook_path in Path(repo_path).glob(glob_pattern):
                    print(f"Processing .ipynb file: {notebook_path}")
                    loader = NotebookLoader(str(notebook_path), include_outputs=True, max_output_length=20, remove_newline=True)
                    for doc in loader.load():
                        notebook_documents.append(doc)
                print(f"Loaded {len(notebook_documents)} .ipynb documents")
                documents += notebook_documents
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)
                loaded_documents = loader.load() if callable(loader.load) else []
                print(f"Loaded {len(loaded_documents)} documents with pattern '{glob_pattern}'")
                documents += loaded_documents

            if documents:
                file_type_counts[ext] = len(documents)
            for doc in documents:
                file_path = doc.metadata['source']
                relative_path = os.path.relpath(file_path, repo_path)
                file_id = str(uuid.uuid4())  # Generate a unique file identifier as a string
                doc.metadata['source'] = relative_path
                doc.metadata['file_id'] = file_id  # Add the file identifier to the metadata
                filenames.append(relative_path)  # Add the filename to the list

                # Add the document to the dictionary using the file_id as the key
                documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Split each document in the dictionary and store them in a new dictionary with the same file_ids
    split_documents_dict = {}
    for file_id, original_doc in documents_dict.items():
        print(f"Processing file_id: {file_id}")  # Add this print statement
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            file_id = split_doc.metadata['file_id']
            original_doc = documents_dict[file_id]
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']

        split_documents_dict[file_id] = split_docs


    # Flatten the split_documents_dict into a list
    documents = [split_doc for split_docs in split_documents_dict.values() for split_doc in split_docs]

    # Add the file_id metadata field to the documents after splitting
    for i, (split_doc, original_doc) in enumerate(zip(documents, documents_dict.values())):
        split_doc.metadata['file_id'] = original_doc.metadata['file_id']

    embeddings = OpenAIEmbeddings()
    index = None
    if documents:
        index = Chroma.from_documents(documents, embeddings)
    return index, documents, file_type_counts, filenames  # Updated the return statement to include filenames


def ask_question(similar_docs, question, llm_chain, repo_name, github_url, conversation_history, file_type_counts, filenames):
    numbered_documents = format_documents(similar_docs)
    file_type_summary = ", ".join([f"{count} {ext} files" for ext, count in file_type_counts.items()])
    file_name_summary = ", ".join([f"{i+1}. {filename}" for i, filename in enumerate(filenames)])
    context = f"This question is about the GitHub repository '{repo_name}' available at {github_url}. The repository contains {file_type_summary}. Some files are split into smaller pieces for better processing. The actual files in the repository are: {file_name_summary}"
    answer_with_sources = llm_chain.run(
        question=question,
        context=context,
        numbered_documents=numbered_documents,
        repo_name=repo_name,
        github_url=github_url,
        conversation_history=conversation_history
    )
    return answer_with_sources

if __name__ == "__main__":
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    repo_link = github_url
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)  # Updated the function call to receive filenames

            if index is None:
                print("No documents were found to index. Exiting.")
                exit()

            print("Repository cloned. Indexing files...")
            llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
            template = """
                Conversation History:
                {conversation_history}
                Current Question: {question}

                Repository Details:
                - Name: {repo_name}
                - URL: {github_url}

                Context: {context}

                Codebase Information:
                {numbered_documents}

                Instructions: 
                - Provide a detailed answer (when necessary) to the current question based on the provided documents and context.
                - If you do not need to provide a detailed answer, simply provide a short answer.
                - Refer to the conversation history for context regarding previous questions and answers.
                - If you cannot find a relevant answer, reply with 'I am not sure'.
                - Do not make up any information; only work with the context and documents provided.
                - MANDATORY: Follow instructions exactly as they are written above. 

                Answer:
            """
            prompt = PromptTemplate(
                template=template,
                input_variables=["conversation_history", "question", "context", "numbered_documents", "repo_name", "github_url"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            conversation_history = ""
            while True:
                try:
                    user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == "exit()":
                        break
                    k = min(4, len(documents))
                    similar_docs = index.similarity_search(user_question, k=k)

                    if similar_docs:
                        numbered_documents = format_documents(similar_docs)
                        answer = ask_question(similar_docs, user_question, llm_chain, repo_name, github_url, conversation_history, file_type_counts, filenames)
                        print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                        conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
            print("Failed to clone the repository.")
