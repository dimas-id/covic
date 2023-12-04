from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='main/static/data/collections',
                          postings_encoding=VBEPostings,
                          output_dir='main/static/data/index')

# queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri",
#            "Terletak sangat dekat dengan khatulistiwa"]
# queries = ["Terletak sangat dekat dengan khatulistiwa"]
# queries = ['Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri, sebelum perusahaan asuransi atau (majikan yang diasuransikan sendiri) akan membayar 100 persen untuk biaya perawatan kesehatan individu.']
# queries = ['Pola angin global: Angin diberi nama berdasarkan arah dari mana mereka bertiup. Dunia dikelilingi oleh enam sabuk angin utama, tiga di setiap belahan bumi. Dari kutub ke khatulistiwa, mereka adalah kutub timur, barat, dan angin pasat. Keenam sabuk bergerak ke utara di musim panas utara dan ke selatan di musim dingin utara. Angin: Di selatan sekitar 30 derajat, angin pasat utara atau timur laut bertiup sebagian besar dari timur laut menuju khatulistiwa. Ini adalah angin favorit pelaut, karena cuacanya hangat, dan angin biasanya bertiup terus ke arah yang menguntungkan. Columbus menggunakan ini untuk berlayar ke Karibia.']
# queries = ['makan minum tidur ulangi']
# queries = ['Jawabannya: Barang siapa yang bunuh diri dengan mencekik akan tetap mencekik dirinya sendiri di dalam Api Neraka (selamanya) dan siapa yang bunuh diri dengan menusuk dirinya sendiri akan tetap menusuk dirinya sendiri di dalam Api Neraka. (Bukhari, Janaiz 84). Bunuh diri adalah dosa besar yang dilarang oleh Islam. Seorang Muslim membunuh dirinya sendiri adalah pembunuhan dan dosa yang lebih besar daripada membunuh orang lain. Oleh karena itu, para cendekiawan Islam memperdebatkan apakah shalat jenazah orang yang bunuh diri dilakukan atau tidak. Diskusi ini tidak ditujukan untuk seorang pembunuh yang membunuh seseorang. Doa pemakaman seorang pembunuh dilakukan. Muslim membunuh dirinya sendiri adalah pembunuhan dan dosa yang lebih besar daripada membunuh orang lain. Dengan demikian, para ulama Islam memperdebatkan apakah shalat jenazah orang yang bunuh diri dilakukan atau tidak. Diskusi ini tidak dibuat untuk seorang pembunuh yang membunuh seseorang.']
# queries = ['Kolesistitis kronis terjadi setelah episode kolesistitis akut berulang dan hampir selalu disebabkan oleh batu empedu']
# queries = ['the psychological restructuring having been disorganized in connection with the disease but also with the hospital and family environment, strongly impacted by anxiety.']
# queries = ['Spongiform degeneration is characterized by vacuolation in nervous tissue accompanied by neuronal death and gliosis. Although spongiform degeneration is a hallmark of prion diseases, this pathology is also present in the brains of patients suffering from Alzheimer\'s disease, diffuse Lewy body disease, human immunodeficiency virus (HIV) infection, and Canavan\'s spongiform leukodystrophy. The shared outcome of spongiform degeneration in these diverse diseases suggests that common cellular mechanisms must underlie the processes of spongiform change and neurodegeneration in the central nervous system. Immunohistochemical analysis of brain tissues reveals increased ubiquitin immunoreactivity in and around areas of spongiform change, suggesting the involvement of ubiquitin-proteasome system dysfunction in the pathogenesis of spongiform neurodegeneration. The link between aberrant ubiquitination and spongiform neurodegeneration has been strengthened by the discovery that a null mutation in the E3 ubiquitin-protein ligase mahogunin ring finger-1 (Mgrn1) causes an autosomal recessively inherited form of spongiform neurodegeneration in animals. Recent studies have begun to suggest that abnormal ubiquitination may alter intracellular signaling and cell functions via proteasome-dependent and proteasome-independent mechanisms, leading to spongiform degeneration and neuronal cell death. Further elucidation of the pathogenic pathways involved in spongiform neurodegeneration should facilitate the development of novel rational therapies for treating prion diseases, HIV infection, and other spongiform degenerative disorders.']
queries = ['covid','HIV']
# queries = ['apa artinya']
for query in queries:
    print("Query  : ", query)
    print("Results:")
    # for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
