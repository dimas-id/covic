import ir_datasets
import pickle
import os


# Load the all of the data
def load_data(ir_datasets_link, num_batch=15, num_docs_limit=75000):

    print("\n===== Start Loading Data =====")
    # Dataset and splitting
    dataset = ir_datasets.load(ir_datasets_link)
    docs = dataset.docs_iter()          # Docs
    queries = dataset.queries_iter()    # Queries
    qrels = dataset.qrels_iter()        # Qrels

    load_docs(docs, num_batch, num_docs_limit)
    load_queries(queries)
    load_qrels(qrels)

    print("===== Finished Loading Data =====\n")

# Load the collection docs
def load_docs(docs_data, docs_batches, docs_limit):
    
    batch_count = 0
    limit_counter = 1
    limit = docs_limit
    empty_counter = 0

    print("\nloading docs...")

    while limit_counter <= limit :

        for doc in docs_data:
            
            # Batch foldering
            batch_id = batch_count % docs_batches
            batch_dir = f'collections/{batch_id}'

            # Create the batch directory if it doesn't already exist
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir)

            # Write to each docs
            if (doc.text):
                with open(f'{batch_dir}/{doc.doc_id}.txt', 'w', encoding="utf-8") as f:
                    f.write(doc.text)
            else:
                empty_counter += 1
            
            batch_count += 1
            limit_counter += 1

    print("empty docs:", empty_counter, "out of", docs_limit)
    print("loading docs has done\n")

# Load the queries file
def load_queries(query_data) :
    print("\nloading queries...")
    with open('queries.txt', 'w', encoding="utf-8") as f:
        for query in query_data:
            f.write(f"{query.query_id} {query.text}\n")
    print("loading queries has done\n")

# Load the qrels file
def load_qrels(qrel_data):
    print("\nloading qrels...")
    with open('qrels.txt', 'w', encoding="utf-8") as f:
        for qrel in qrel_data:
            f.write(f"{qrel.query_id} {qrel.doc_id} {qrel.relevance}\n")
    print("loading qrels has done\n")

# Print out queries
def print_queries(query_data):
    for query in query_data:
        print("Id:", query.query_id)
        print("Text:", query.text)
        print("Query:",query.query)
        print("Narrative:",query.narrative, "\n")


if __name__ == "__main__":
    
    load_data('beir/trec-covid', 20, 100000)