# @Author: Ibrahim Bashir
# @Date: 21 November 2026
# @Version: 1
# This program compares keyword, meaning-based and combined search to see which best finds the right Systems-Lab revision topic for a student’s question.

import csv
from pathlib import Path

data_folder = Path(__file__).resolve().parent

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

with open(data_folder / "topics.csv", encoding="utf-8-sig", newline="") as file:
    topics = list(csv.DictReader(file))

def split_words(text):
    text = text.lower().replace("’", "'")
    return text.split()

topic_words = []

for topic in topics:
    text = topic["title"] + " " + topic["description"]
    topic_words.append(split_words(text))

search_index = BM25Okapi(topic_words)
query = input("What are you looking for? ")
scores = search_index.get_scores(split_words(query))

if any(scores):
    results = list(zip(scores, topics))
    results.sort(key=lambda result: result[0], reverse=True)

    for score, topic in results:
        print(f"{score:.2f}  {topic['title']}")
else:
    print("No keyword match found.")

print("\nMeaning-based search results")
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

topic_texts = []
for topic in topics:
    topic_texts.append(topic["title"] + " " + topic["description"])

topic_embeddings = model.encode_document(topic_texts)
query_embedding = model.encode_query(query)
meaning_scores = model.similarity(query_embedding, topic_embeddings)[0].tolist()

meaning_results = list(zip(meaning_scores, topics))
meaning_results.sort(key=lambda result: result[0], reverse=True)

for rank, (score, topic) in enumerate(meaning_results, start=1):
    print(f"{rank}. {topic['title']} ({score:.3f})")

from scipy.stats import rankdata

keyword_ranks = rankdata(-scores)
meaning_ranks = rankdata([-score for score in meaning_scores])
average_ranks = (keyword_ranks + meaning_ranks) / 2

combined_results = list(zip(average_ranks, topics))
combined_results.sort(key=lambda result: result[0])

print("\nCombined search results")
for average, topic in combined_results:
    print(f"{average:.2f}  {topic['title']}")

with open(data_folder / "queries.csv", encoding="utf-8-sig", newline="") as file:
    test_queries = [row for row in csv.DictReader(file) if row["query"].strip()]

print(f"\nLoaded {len(test_queries)} test searches.")
print("First search:", test_queries[0]["query"])
print("Expected topic ID:", test_queries[0]["expected_id"])

topic_ids = [topic["id"].strip() for topic in topics]

print("\nTest results — rank 1 means the expected topic came first")
print("Keyword | Meaning | Combined | Search")

for test in test_queries:
    question = test["query"]
    target = topic_ids.index(test["expected_id"].strip())

    keyword_scores = search_index.get_scores(split_words(question))
    keyword_ranks = rankdata(-keyword_scores)

    question_embedding = model.encode_query(question)
    similarities = model.similarity(question_embedding, topic_embeddings)[0].tolist()
    meaning_ranks = rankdata([-score for score in similarities])

    average_ranks = (keyword_ranks + meaning_ranks) / 2
    combined_ranks = rankdata(average_ranks)

    keyword = f"{keyword_ranks[target]:g}" if any(keyword_scores) else "none"
    print(f"{keyword:>7} | {meaning_ranks[target]:>7g} | "
          f"{combined_ranks[target]:>8g} | {test['type']} — {question}")

input("\nPress Enter to close.")
