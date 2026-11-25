# Systems Search

A Python program that compares keyword, meaning-based and combined search
across five Systems-Lab revision topics.

## Results

On ten test searches, the expected topic ranked first 6 times with keyword
search, 10 times with meaning-based search and 8 times with combined search.

## Run the current source on Windows

The current source downloads safetensors weights from a pinned model revision. In PowerShell, run these commands from the repository folder with Python 3.12 installed.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple "setuptools>=84.1.0"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install sentence-transformers rank-bm25 scipy
.\.venv\Scripts\python.exe search.py
```

## Windows download

The v1.0.0 executable predates the model revision and weight-format checks.

[Download Systems Search v1.0.0 for Windows (.exe)](https://github.com/polarvapourer/Systems-Search/releases/download/v1.0.0/SystemsSearch.exe) · [Release notes](https://github.com/polarvapourer/Systems-Search/releases/tag/v1.0.0)

The first run needs an internet connection to download the search model.