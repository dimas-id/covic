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

    return render(request, 'search_result.html', context)

def detail(request, path):
    context = {}
    