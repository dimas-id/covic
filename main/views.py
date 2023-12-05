from django.shortcuts import render

from main.bsbi import BSBIIndex
from main.compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='main/static/data/collections',
                          postings_encoding=VBEPostings,
                          output_dir='main/static/data/index')

def home(request):
    return render(request, 'home.html')

def search(request):
    context = {}
    if request.method == 'GET':
        query = request.GET.get('q')

        if query:
            res_bm25 = BSBI_instance.retrieve_bm25(
                query, k=100, k1=1.2, b=0.8
            )
            result = []
            for score, doc in res_bm25:
                print(f'score: {score}, doc: {doc}')
                # doc = doc.replace('\\', '/')
                # with open(f'{doc}', 'r', encoding='utf-8') as f:
                #     text = f.read()
                # title = doc.split('/')[-1].replace('.txt','')
                # path = doc.find('collections')

    return render(request, 'result.html', context)

def detail(request, path):
    context = {}
    