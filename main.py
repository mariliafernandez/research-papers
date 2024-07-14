from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from pathlib import Path
from typing import List
import sys
import re
import json
import pymupdf

json_parse_prompt = "Return the answer in a valid JSON format, with the key '{key}'. DO NOT return any more content, but ONLY the JSON string. Remember to escape double quotes by using backslash (\)."


def get_keywords_block(page: pymupdf.Page) -> dict:
    blocks = page.get_text("blocks", sort=True)
    text_blocks = [b[4] for b in blocks]
    for text in text_blocks:
        m = re.search("[K|k]ey\s?[W|w]ords[\.|:].*", text, flags=re.DOTALL)
        if m:
            return (
                re.sub("[K|k]ey\s?[W|w]ords[\.|:]", "", m.group())
                .replace("\n", " ")
                .strip()
            )


def max_block_size(block: dict) -> int:
    sizes = []
    if "lines" in block:
        for l in block["lines"]:
            sizes += [round(s["size"]) for s in l["spans"]]
    if len(sizes) > 0:
        return max(sizes)
    return 0


def get_title_block(page: pymupdf.Page) -> dict:
    blocks = page.get_text("dict", sort=True)["blocks"]
    title_block = blocks[0]
    for b in blocks:
        block_sizes = []
        if "lines" in b:
            for l in b["lines"]:
                if l["dir"] != (1, 0):  # if line not horizontal, ignore it
                    continue
                block_sizes += [round(s["size"]) for s in l["spans"]]
        if len(block_sizes) > 0:
            block_max_size = max(block_sizes)
            if block_max_size > max_block_size(title_block):
                title_block = b
    return title_block


def get_block_text(block: dict) -> str:
    block_text = ""
    if "lines" in block:
        for l in block["lines"]:
            line_text = "".join([s["text"] for s in l["spans"]])
            block_text = " ".join([block_text, line_text])
    return block_text.strip()


def clean_json_string(json_string: str) -> str:
    q = []
    init = json_string.index("{")
    end = len(json_string) - json_string[::-1].index("}")

    for i in range(init, init + len(json_string[init:end])):
        char = json_string[i]
        if char == "{":
            q.append(char)
        elif char == "}":
            q.pop()
            if len(q) == 0:
                return json_string[init : i + 1]
    return json_string[init:end]


def build_search_string(title: str, keywords: str) -> str:
    task_prompt = "Here is the title and keywords of a research paper: \nTitle: {title}\n\nKeywords:{keywords}.\n\nBuild a search string to use in a search engine for research papers in order to find relevant papers related to this specific article."

    llm = Ollama(model="llama3")

    prompt = (
        task_prompt.format(title=title, keywords=keywords)
        + "\n\n"
        + json_parse_prompt.format(key="search_string")
    )

    search_string = llm.invoke(prompt)

    cleaned = clean_json_string(search_string)
    data = json.loads(cleaned)

    return data["search_string"].replace('"', "")


def arxiv_search(search_string: str, max_results: int) -> List[Document]:
    arxiv = ArxivAPIWrapper(
        top_k_results=max_results,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=max_results,
        load_all_available_meta=False,
        doc_content_chars_max=40000,
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


if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    for filepath in input_path.glob("*.pdf"):
        doc = pymupdf.open(filepath)
        title = get_block_text(get_title_block(doc[0]))
        keywords = get_keywords_block(doc[0])

        print("filename:", filepath.name)
        print("title:", title)
        print("keywords:", keywords)
        print("search string:", build_search_string(title, keywords))
        print()
