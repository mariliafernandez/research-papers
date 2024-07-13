from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.documents import Document
from langchain_community.llms import Ollama
import json
from typing import List

def build_search_string(theme:str) -> str:
    prompt_command = f"Build a search string to use in a search engine for research papers, based on the following theme: '{theme}'."
    prompt_format_json = "\nReturn the answer in a valid JSON format, with the key \"search_string\". DO NOT return any more content, but ONLY the JSON string. Remember to escape double quotes by using backslash (\\)."

    llm = Ollama(model="llama3")

    prompt = prompt_command + prompt_format_json
    search_string = llm.invoke(prompt)
    data=json.loads(search_string)
    return data['search_string'].replace('"', '')


def arxiv_search(search_string:str, max_results:int) -> List[Document]:
    arxiv = ArxivAPIWrapper(
        top_k_results = max_results,
        ARXIV_MAX_QUERY_LENGTH = 300,
        load_max_docs = max_results,
        load_all_available_meta = False,
        doc_content_chars_max = 40000
    )

    results = arxiv.arxiv_search(search_string, max_results=max_results).results()

    return [
        Document(
            page_content=result.summary,
            metadata={
                "Entry ID": result.entry_id,
                "Published": result.updated.date(),
                "Title": result.title,
                "Authors": ", ".join(a.name for a in result.authors),
            },
        )
        for result in results
    ]

