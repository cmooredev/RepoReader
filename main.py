#main.py
import os
import tempfile
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from config import WHITE, GREEN, RESET_COLOR, model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files, is_repo_cloned as _is_repo_cloned, extract_repo_name
from questions import ask_question, QuestionContext

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IS_TEMP_DIR = False

def main():
    github_url = input("Enter the GitHub URL of the repository: ")
    # github_url = r"https://github.com/Lightricks/dwh-data-model-transforms"
    repo_name = extract_repo_name(github_url)
    print("Cloning the repository...")


    with tempfile.TemporaryDirectory() as local_path:

        if not IS_TEMP_DIR:
            local_path = 'stat_path_repos'

            # check if the repo is already cloned
            is_repo_cloned = _is_repo_cloned(github_url, local_path)
            print(f'[LOG] is repo {repo_name} already cloned? {is_repo_cloned}')

            # if the repo is already cloned in the static path, then skip cloning. If not, clone it in the static path
            repo_condition = clone_github_repo(github_url, os.path.join(local_path, repo_name)) if not is_repo_cloned else True
        else:
            repo_condition = clone_github_repo(github_url, local_path)

        print(f'[LOG] repo_condition: {repo_condition}')

        if repo_condition:
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                print("No documents were found to index. Exiting.")
                exit()

            print("Repository cloned. Indexing files...")
            llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.2)

            template = """
            Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

            Instr:
            1. Answer based on context/docs.
            2. Focus on repo/code.
            3. Consider:
                a. Purpose/features - describe.
                b. Functions/code - provide details/samples.
                c. Setup/usage - give instructions.
            4. Unsure? Say "I am not sure".

            Answer:
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents", "file_type_counts", "filenames"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            conversation_history = ""
            question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
            while True:
                try:
                    user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == "exit()":
                        break
                    print('Thinking...')
                    user_question = format_user_question(user_question)

                    answer = ask_question(user_question, question_context)
                    print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                    conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
            print("Failed to clone the repository.")
