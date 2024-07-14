# (re)search papers
Tool for searching related papers on arxiv based on source PDF paper


## How does it work
1. Extracts the title, keywords and abstract from the input PDF 
2. Generates a search string based on the extracted context using LLM
3. Retrieves results from the arXiv API using the created search string

## Technologies
* [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/the-basics.html)
* [Langchain](https://www.langchain.com/)
* [Llama3](https://ollama.com/library/llama3)
* [arXiv API](https://info.arxiv.org/help/api/index.html)