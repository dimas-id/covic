import os
from tqdm import tqdm

def merge_files(collections_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as output:
        for folder in tqdm(os.listdir(collections_path)):
            batch = os.path.join(collections_path, folder)
            for file in tqdm(os.listdir(batch)):
                file_path = os.path.join(batch, file)
                with open(file_path, 'r', encoding='utf-8') as input_file:
                    content = input_file.read().strip()
                    doc_id = os.path.splitext(file)[0]
                    output.write(f'{doc_id} {content}\n')

# if __name__ == 'main':
current_directory = os.path.dirname(os.path.abspath(__file__))
# print(os.path.join(current_directory, 'collections','0'))
collections_path = os.path.join(current_directory, 'collections')
output_file = os.path.join(current_directory, 'docs.txt')
# print(collections_path)
# print(output_file)

merge_files(collections_path, output_file)
print('berhasil')