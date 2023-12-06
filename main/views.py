import os
from django.shortcuts import render

from main.bsbi import BSBIIndex
from main.compression import VBEPostings
from main.letor import Letor
from tqdm import tqdm

from django.shortcuts import redirect
from django.core.paginator import Paginator

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='main/static/data/collections',
                          postings_encoding=VBEPostings,
                          output_dir='main/static/data/index')
current_directory = os.path.dirname(os.path.abspath(__file__))
train_docs_path = os.path.join(current_directory,"static/data/docs.txt")
train_queries_path = os.path.join(current_directory,"static/data/queries.txt")
train_qrels_path = os.path.join(current_directory,"static/data/qrels.txt")

model_path = os.path.join(current_directory, 'static','model', 'ranker_model.pkl')

letor = Letor(train_docs_path, train_queries_path, train_qrels_path)
letor.load_model(model_path=model_path)

def home(request):
    return render(request, 'home.html')

def search(request):
    context = {}
    if request.method == 'GET':
        query = request.GET.get('q')
        # print(f'Q: {query}')

        if query:
            rank_bm25 = []
            serp_bm25 = BSBI_instance.retrieve_bm25(
                query, k=100, k1=1.2, b=0.8
            )
            result = []
            docs_bm25 = []
            for score, doc in serp_bm25:
                doc = doc.replace('\\', '/')
                docid = os.path.splitext(os.path.basename(doc))[0]
                str_content = ''
                with open(doc, encoding='utf-8') as file:
                    content = file.read()
                # docs_bm25.append((docid,content))
                    docs_bm25.append((doc,content))
            
            rankings_bm25_letor = []
            queries_bm25 = query
            
            if len(docs_bm25) != 0:
                rankings_bm25_letor = letor.predict_rankings(queries_bm25, docs_bm25)

            for (doc, score) in rankings_bm25_letor:
                # print('Iterate serp bm25_letor')
                doc = doc.replace('\\', '/')
                with open(f'{doc}', 'r', encoding='utf-8') as file:
                    text = file.read()

                did = os.path.splitext(os.path.basename(doc))[0]
                title_doc = did
                idx = doc.find('collections/')
                path = doc[idx:]
                # print(f'score: {score}, doc: {doc}')

                result.append((title_doc, f'{text[:250]} ...', path))
            
            paginator = Paginator(result, 10)

            context['result'] = paginator.get_page(request.GET.get('page'))
            context['query'] = query

            return render(request, 'search_result.html', context)

    # return render(request, 'home.html')
    return redirect('main:home')

def detail(request, path):
    context = {}
    title = path.split('/')[-1].replace('.txt', '')
    with open(f'main/static/data/{path}', 'r', encoding='utf-8') as file:
        text = file.read()
    context['title'] = title
    context['text'] = text
    return render(request, 'detail_doc.html', context)
    
def about(request):
    return render(request, 'about.html')

def devs(request):
    return render(request, 'devs.html')