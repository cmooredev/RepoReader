import os
from pathlib import Path
import subprocess
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
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
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig']
    
    documents_list = []
    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = DirectoryLoader(repo_path, glob=glob_pattern)
            if callable(loader.load): 
                documents = loader.load()
                for doc in documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    doc.metadata['source'] = relative_path
                documents_list.extend(documents)
            else: 
                print(f"Loader does not have a callable load method for pattern: {glob_pattern}")
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents_list)
    embeddings = OpenAIEmbeddings()
    index = None
    if documents:
        index = Chroma.from_documents(documents, embeddings)
    return index, documents

def ask_question(similar_docs, question, llm_chain, repo_name, github_url, conversation_history):
    numbered_documents = format_documents(similar_docs)
    context = f"This question is about the GitHub repository '{repo_name}' available at {github_url}."
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
            index, documents = load_and_index_files(local_path)

            if index is None:
                print("No documents were found to index. Exiting.")
                exit()

            print("Repository cloned. Indexing files...")
            llm = OpenAI(api_key=OPENAI_API_KEY)
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
                        answer = ask_question(similar_docs, user_question, llm_chain, repo_name, github_url, conversation_history)
                        print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                        conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
            print("Failed to clone the repository.")
