from utils import clean_and_tokenize, format_documents



class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames

def ask_question(question, context: QuestionContext):
    tokenized_question = clean_and_tokenize(question)
    scores = context.index.get_scores(tokenized_question)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    similar_docs = [context.documents[i] for i in top_k_indices]
    relevant_docs = [doc for doc in similar_docs if len(set(tokenized_question).intersection(clean_and_tokenize(doc.page_content))) >= 3]

    numbered_documents = format_documents(relevant_docs)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}"

    answer_with_sources = context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context,
        repo_name=context.repo_name,
        github_url=context.github_url,
        conversation_history=context.conversation_history,
        numbered_documents=numbered_documents
    )
    return answer_with_sources
