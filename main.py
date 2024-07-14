from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from pathlib import Path
from typing import List
import sys
import re
import json
import pymupdf
from arxiv.arxiv import SortCriterion

json_parse_prompt = "Return the answer in a valid JSON format, with the key '{key}'. DO NOT return any more content, but ONLY the JSON string. Remember to escape double quotes by using backslash (\)."


def get_text_block_match(regex_str: str, text_blocks: List[str]) -> str:
    separator_regex = "[\.|:]?"

    for i in range(len(text_blocks)):
        m = re.search(
            regex_str + separator_regex, text_blocks[i].lower(), flags=re.DOTALL
        )
        if m:
            return i, m.start(), m.end()


def get_key_value(text_blocks: List[str], i: int, key_start: int, key_end: int) -> str:
    text = text_blocks[i]
    if text[key_end:].replace("\n", "").strip() == "":
        text = text_blocks[i + 1]
    else:
        text = text[key_end:]
    s = re.search("[A-Za-z].*", text, flags=re.DOTALL)
    if s:
        return text[s.start() :]
    return text[key_start:]


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


def build_search_string(title: str, abstract: str, keywords: str) -> str:
    task_prompt = """Here are the title, abstract and keywords of a research paper:
    Title: {title}
    Abstract: {abstract}
    Keywords: {keywords}

    Your job is to build a search string to use in a search engine for research papers, in order to find relevant papers related to this specific one, following the steps:
    1. Read the keywords and select the relevant ones based on the context of the title and the abstract.
    2. If necessary you can add more keywords related to the subject to expand the search.
    3. Use the selected keywords and combine them by using the operators AND and OR to get more related results and expand the search."""
    llm = Ollama(model="llama3")

    prompt = (
        task_prompt.format(title=title, abstract=abstract, keywords=keywords)
        + "\n\n"
        + json_parse_prompt.format(key="search_string")
    )

    search_string = llm.invoke(prompt)
    cleaned = clean_json_string(search_string)
    return cleaned


def arxiv_search(
    search_string: str, max_results: int, sorting="relevance"
) -> List[Document]:
    arxiv = ArxivAPIWrapper(
        top_k_results=max_results,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=max_results,
        load_all_available_meta=False,
        doc_content_chars_max=40000,
    )

    sort_by = {
        "relevance": SortCriterion.Relevance,
        "submitted-date": SortCriterion.SubmittedDate,
        "last-updated": SortCriterion.LastUpdatedDate,
    }

    results = arxiv.arxiv_search(
        search_string, max_results=max_results, sort_by=sort_by[sorting]
    ).results()

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


def safe_read_search_str(json_str: str) -> str:
    try:
        return json.loads(json_str)["search_string"].replace('"', "").strip()
    except:
        match = re.search('"search_string":\s?".*"', json_str, flags=re.DOTALL)
        if match:
            return (
                match.group().replace('"search_string":', "").replace('"', "").strip()
            )
        raise Exception("Error parsing JSON string")


def parse_doc(doc):
    return {
        "Summary": doc.page_content,
        "Entry ID": doc.metadata["Entry ID"],
        "Published": str(doc.metadata["Published"]),
        "Title": doc.metadata["Title"],
        "Authors": doc.metadata["Authors"],
    }


def extract_features(first_page) -> dict:
    title = get_block_text(get_title_block(first_page))
    blocks = first_page.get_text("blocks", sort=True)
    text_blocks = [b[4] for b in blocks]

    keywords_i, keywords_start, keywords_end = get_text_block_match(
        "key\s?words", text_blocks
    )
    keywords_value = get_key_value(
        text_blocks, keywords_i, keywords_start, keywords_end
    )

    abstract_i, abstract_start, abstract_end = get_text_block_match(
        "abstract", text_blocks
    )
    abstract_value = get_key_value(
        text_blocks, abstract_i, abstract_start, abstract_end
    )

    return {"title": title, "keywords": keywords_value, "abstract": abstract_value}


def search_pipeline(filepath):
    doc = pymupdf.open(filepath)
    features = extract_features(first_page=doc[0])

    search_json_string = build_search_string(
        title=features["title"],
        abstract=features["abstract"],
        keywords=features["keywords"],
    )
    search_string = safe_read_search_str(search_json_string)
    related = arxiv_search(search_string=search_string, max_results=5)
    return related


if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    output_path = Path("./output")
    output_path.mkdir(exist_ok=True)

    for filepath in input_path.glob("*.pdf"):
        doc = pymupdf.open(filepath)
        features = extract_features(doc[0])
        
        search_json_string = build_search_string(
            title=features["title"],
            abstract=features["abstract"],
            keywords=features["keywords"],
        )

        search_string = safe_read_search_str(search_json_string)
        related_papers = arxiv_search(search_string=search_string, max_results=5)

        output_obj = {
            "title": features["title"],
            "abstract": features["abstract"],
            "keywords": features["keywords"],
            "search_string": search_string,
            "papers": [parse_doc(doc) for doc in related_papers],
        }
        print()

        with open(output_path / f"{filepath.stem}.json", "w") as fp:
            json.dump(output_obj, fp)
