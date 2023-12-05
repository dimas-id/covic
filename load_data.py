import ir_datasets
import pickle
import os


# Load the all of the data
def load_data(ir_datasets_link, num_batch=20, num_docs_limit=-1):

    print("\n===== Start Loading Data =====\n")
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

    print("loading docs...")

    for doc in docs_data:

        if limit_counter == limit:
            break
        
        # Batch foldering
        batch_id = batch_count % docs_batches
        batch_dir = f'collections/{batch_id}'

        # Create the batch directory if it doesn't already exist
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)

        # Write to each docs
        # if (doc.text): # for beir
        if (doc.abstract): # for cord19
            with open(f'{batch_dir}/{doc.doc_id}.txt', 'w', encoding="utf-8") as f:
                # f.write(doc.text) # for beir
                f.write(doc.abstract) # for cord19
        else:
            empty_counter += 1
        
        batch_count += 1
        limit_counter += 1

    print("empty docs:", empty_counter, "out of", limit_counter)
    print("loading docs has done\n")

# Load the queries file
def load_queries(query_data) :
    print("loading queries...")
    with open('queries.txt', 'w', encoding="utf-8") as f:
        for query in query_data:
            # f.write(f"{query.query_id} {query.text}\n") # for beir
            f.write(f"{query.query_id} {query.description}\n") # for cord19
    print("loading queries has done\n")

# Load the qrels file
def load_qrels(qrel_data):
    print("loading qrels...")
    with open('qrels.txt', 'w', encoding="utf-8") as f:
        for qrel in qrel_data:
            f.write(f"{qrel.query_id} {qrel.doc_id} {qrel.relevance}\n")
    print("loading qrels has done\n")

# Print out queries
def print_queries(query_data):
    id_counter = 0
    for query in query_data:
        # teks = query.text

        # if ("fever" in teks.lower()):
        #     print("Id:", query.query_id)
        #     print("Text:", query.text)
        
        # query_id, text, description, narrative, disclaimer, stance, evidence
        print(id_counter)
        print("Id:", query.query_id)
        print("Text:", query.text)
        # print("Query:", query.query)
        # print("Title:", query.title)
        # print("Description:", query.description)
        # print("Narrative:", query.narrative)
        # print("Narrative:", query.narrative)
        # print("Disclaimer:", query.disclaimer)
        # print("Stance:", query.stance)
        # print("Evidence:", query.evidence)
        print("")
        id_counter += 1

# Load 200 queries from cord19 round 1 2 3 4 5
def load_cord_queries():

    print("loading cord queries...")

    queries_list = []
    
    queries_list.append(ir_datasets.load('cord19/trec-covid/round1').queries_iter())
    queries_list.append(ir_datasets.load('cord19/trec-covid/round2').queries_iter())
    queries_list.append(ir_datasets.load('cord19/trec-covid/round3').queries_iter())
    queries_list.append(ir_datasets.load('cord19/trec-covid/round4').queries_iter())
    queries_list.append(ir_datasets.load('cord19/trec-covid/round5').queries_iter())

    with open('queries_cord.txt', 'w', encoding="utf-8") as f:
        id_counter = 1
        for query_instance in queries_list:
            for query in query_instance:
                f.write(f"{id_counter} {query.description}\n") # for cord19
                id_counter += 1
    
    print("loading cord queries has done\n")

# Load 200 queries from cord19 round 1 2 3 4 5
def load_final_queries():

    print("loading cord queries...")

    queries_list = []
    
    queries_list.append(ir_datasets.load('beir/trec-covid').queries_iter())
    queries_list.append(ir_datasets.load('cord19/trec-covid').queries_iter())

    with open('queries_final.txt', 'w', encoding="utf-8") as f:
        id_counter = 1
        for query_instance in queries_list:
            for query in query_instance:
                try:
                    f.write(f"{id_counter} {query.description}\n") # for cord19
                except: 
                    f.write(f"{id_counter} {query.text}\n") # for beir

                id_counter += 1
    
    print("loading cord queries has done\n")


if __name__ == "__main__":
    
    # data_link = 'beir/trec-covid'
    # data_link = 'cord19/trec-covid'
    # data_link = "clinicaltrials/2021/trec-ct-2022"
    # data_link = "c4/en-noclean-tr/trec-misinfo-2021"
    # data_link = "nfcorpus/dev/nontopic"
    data_link = "nfcorpus/train/nontopic"

    # load_data(ir_datasets_link=data_link)

    # load_cord_queries()
    # load_final_queries()

    # dataset = ir_datasets.load(data_link)
    # queries = dataset.queries_iter()
    # print(type(queries))

    dataset = ir_datasets.load(data_link)
    queries = dataset.queries_iter()
    print_queries(queries)
