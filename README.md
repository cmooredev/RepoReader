# Code Repository Explorer

Explore and ask questions about a GitHub code repository using OpenAI's GPT-3 language model.

## Prerequisites

- Python 3.6+
- OpenAI API key (set in the environment variable `OPENAI_API_KEY`)

## Usage
1. Set the OpenAI API key as an environment variable `OPENAI_API_KEY`.
2. Run the script: `reporeader.py`
3. Enter the GitHub URL of the repository to explore.
4. Ask questions about the repository. Type `exit()` to quit.

## Key Features
- Clones and indexes the contents of a GitHub repository.
- Supports various file types, including code, text, and Jupyter Notebook files.
- Generates detailed answers to user queries based on the repository's contents.
- Uses OpenAI's language model for generating responses.
- Supports interactive conversation with the language model.
- Presents top relevant documents for each question.
