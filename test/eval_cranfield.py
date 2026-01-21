import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

import ir_datasets
import math
import statistics
from src import schemas
from src.core import config
from src.schemas import IRDatasetIngestionRequest
from src.services.public import ingest_ir_dataset, retrieve_documents
from eval_utils import check_table_exists

DATASET_NAME = "cranfield"


def get_queries_dict(dataset_name: str):
    ds = ir_datasets.load(dataset_name)
    queries = list(ds.queries_iter())
    qrels = list(ds.qrels_iter())

    queries_dict: dict[str, dict[str, any]] = {}
    for query in queries:
        queries_dict[str(query.query_id)] = {
            "text": query.text.replace("\n", " ").strip(),
            "relevant_docs": {},
        }
    for qrel in qrels:
        query_id = qrel.query_id
        doc_id = qrel.doc_id
        if query_id in queries_dict:
            queries_dict[query_id]["relevant_docs"][doc_id] = qrel.relevance

    return queries_dict


if __name__ == "__main__":
    table_existed = check_table_exists(DATASET_NAME)

    if table_existed:
        print(f"Table for dataset '{DATASET_NAME}' exists in the database.")
    else:
        print(
            f"Table for dataset '{DATASET_NAME}' does NOT exist in the database. Create a new one..."
        )

        req = IRDatasetIngestionRequest(dataset_name=DATASET_NAME)
        res = ingest_ir_dataset(req)

        if res.status == 201:
            print(f"Successfully ingested dataset '{DATASET_NAME}' into the database.")
        else:
            raise Exception(f"Failed to ingest dataset '{DATASET_NAME}': {res.message}")

    queries_dict = get_queries_dict(DATASET_NAME)
    print(f"Total queries in dataset '{DATASET_NAME}': {len(queries_dict)}")

    retrieval_request = schemas.RetrievalRequest(
        collection_name=DATASET_NAME,
        queries=[q["text"] for q in queries_dict.values()],
        mode="hybrid",
        top_k=10,
        sparse_search_method="inverted_index",
        rerank_enabled=False,
    )

    print("Starting document retrieval for evaluation...")
    retrieval_response = retrieve_documents(retrieval_request)

    if retrieval_response.status == 200:
        results = retrieval_response.results
        print(f"Successfully retrieved documents for {len(results)} queries.")

        # Initialize lists to store metrics for every query
        precisions, recalls, f1_scores = [], [], []
        average_precisions = []  # For MAP
        ndcgs = []  # For NDCG

        # Store detailed info for case studies
        per_query_details = []

        # We need the query IDs in the same order they were sent to retrieval
        query_ids = list(queries_dict.keys())
        k = retrieval_request.top_k  # Top-k

        for i, retrieved_docs in enumerate(results):
            query_id = query_ids[i]
            query_text = queries_dict[query_id]["text"]

            # Ground truth: dictionary of {doc_id: relevance_score}
            # We filter for relevance > 0
            ground_truth = {
                did: score
                for did, score in queries_dict[query_id]["relevant_docs"].items()
                if score > 0
            }

            # If no relevant documents exist for this query, skip it
            if not ground_truth:
                continue

            retrieved_ids = [doc.id for doc in retrieved_docs]
            retrieved_at_k = retrieved_ids[:k]

            # --- 1. Unranked Metrics (P, R, F1) ---
            relevant_retrieved = [doc for doc in retrieved_at_k if doc in ground_truth]

            num_retrieved = len(retrieved_at_k)
            num_relevant_retrieved = len(relevant_retrieved)
            num_total_relevant = len(ground_truth)

            p = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0

            r = (
                num_relevant_retrieved / num_total_relevant
                if num_total_relevant > 0
                else 0
            )

            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

            # --- 2. Ranked Metric: MAP@k ---
            hits = 0
            sum_precisions = 0

            for rank, doc_id in enumerate(retrieved_at_k, start=1):
                if doc_id in ground_truth:
                    hits += 1
                    sum_precisions += hits / rank

            # AP@k (your definition): (1 / |Rel|) * sum_{i=1..k} P@i * r_i
            ap = sum_precisions / num_total_relevant if num_total_relevant > 0 else 0
            average_precisions.append(ap)

            # --- 3. Ranked Metric with Scores: NDCG@k ---
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved_at_k, start=1):
                if doc_id in ground_truth:
                    rel_score = ground_truth[doc_id]
                    dcg += rel_score / math.log2(rank + 1)

            ideal_scores = sorted(ground_truth.values(), reverse=True)
            ideal_top_k = ideal_scores[:k]

            idcg = 0.0
            for rank, score in enumerate(ideal_top_k, start=1):
                idcg += score / math.log2(rank + 1)

            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)

            # --- Store details for Case Study ---
            per_query_details.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "metrics": {
                        "NDCG": ndcg,
                        "MAP": ap,
                        "F1": f1,
                        "Precision": p,
                        "Recall": r,
                    },
                    "retrieved": retrieved_docs,
                    "ground_truth_ids": list(ground_truth.keys()),
                }
            )

        # --- Aggregate Statistics ---
        mean_p = statistics.mean(precisions)
        mean_r = statistics.mean(recalls)
        mean_f1 = statistics.mean(f1_scores)
        mean_map = statistics.mean(average_precisions)
        mean_ndcg = statistics.mean(ndcgs)

        print("\n" + "=" * 30)
        print(f"EVALUATION RESULTS (Top-k={k})")
        print("=" * 30)
        print(f"Precision@{k}: {mean_p:.4f}")
        print(f"Recall@{k}:    {mean_r:.4f}")
        print(f"F1@{k}:        {mean_f1:.4f}")
        print(f"MAP@{k}:       {mean_map:.4f}")
        print(f"NDCG@{k}:      {mean_ndcg:.4f}")
        print("=" * 30)

        # --- CASE STUDIES ---
        # Sort queries by NDCG (descending)
        sorted_details = sorted(
            per_query_details, key=lambda x: x["metrics"]["NDCG"], reverse=True
        )

        best_cases = sorted_details[:3]  # Top 3
        worst_cases = sorted_details[-3:]  # Bottom 3 (often have 0.0 metrics)

        def print_case_study(title, cases):
            with open("results/cranfield_case_studies.txt", "a") as f:
                f.write(f"\n>>> {title} <<<\n")
                for case in cases:
                    m = case["metrics"]
                    f.write("-" * 60 + "\n")
                    f.write(f"Query ID: {case['query_id']}\n")
                    f.write(f"Query: \"{case['query_text']}\"\n")
                    f.write(
                        f"Metrics: NDCG={m['NDCG']:.4f} | MAP={m['MAP']:.4f} | F1={m['F1']:.4f}\n"
                    )
                    f.write("Retrieved Documents:\n")

                    gt_ids = case["ground_truth_ids"]

                    for rank, doc in enumerate(case["retrieved"], 1):
                        # Check if relevant
                        is_relevant = doc.id in gt_ids
                        status_mark = "[MATCH]" if is_relevant else "[NOT MATCH]"

                        # Truncate title/text for display
                        doc_title = doc.payload.metadata.title.strip() or "(No Title)"
                        doc_snippet = doc.payload.text.replace("\n", " ").strip()

                        f.write(
                            f"  {rank}. {status_mark} [ID: {doc.id}] (Score: {doc.score:.4f})\n"
                        )
                        f.write(f"     Title: {doc_title}\n")
                        f.write(f"     Text:  {doc_snippet}\n")

                    f.write(f"\nTotal Relevant Docs in Dataset: {len(gt_ids)}\n")
                    missing = [
                        did
                        for did in gt_ids
                        if did not in [d.id for d in case["retrieved"]]
                    ]
                    if len(missing) > 5:
                        f.write(f"Missing IDs (first 5): {missing[:5]}...\n")
                    else:
                        f.write(f"Missing IDs: {missing}\n")
                    f.write("-" * 60 + "\n")

        print_case_study("BEST CASES (Highest NDCG)", best_cases)
        print_case_study("WORST CASES (Lowest NDCG)", worst_cases)

        # --- Save to File ---
        os.makedirs("results", exist_ok=True)
        with open("results/cranfield_evaluation_results.txt", "a") as f:
            f.write("EVALUATION RESULTS with config:\n")
            f.write(f"\t+ top-k={k}\n")
            f.write(
                f"\t+ search_mode={retrieval_request.mode}, "
                f"sparse_method={retrieval_request.sparse_search_method}\n"
            )
            f.write(
                f"\t+ fusion_method={config.FUSION_METHOD}, "
                f"fusion_alpha={config.FUSION_ALPHA}\n"
            )
            f.write(f"\t+ rerank_enabled={retrieval_request.rerank_enabled}\n")

            f.write("=" * 30 + "\n")
            f.write(f"Precision@{k}: {mean_p:.4f}\n")
            f.write(f"Recall@{k}:    {mean_r:.4f}\n")
            f.write(f"F1@{k}:        {mean_f1:.4f}\n")
            f.write(f"MAP@{k}:       {mean_map:.4f}\n")
            f.write(f"NDCG@{k}:      {mean_ndcg:.4f}\n")
            f.write("=" * 30 + "\n\n\n")

    else:
        raise Exception(f"Failed to retrieve documents.")
