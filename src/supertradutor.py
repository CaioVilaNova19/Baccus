from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os
import re
import torch

# Caminho do seu modelo fine-tunado
model_path = "../Baccus_en_pt"

print("Carregando modelo fine-tunado...")
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Configurar dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modelo carregado no dispositivo {device}.")

def chunk_text(text):
    max_length = tokenizer.model_max_length
    print(f"Token limit do modelo: {max_length}")

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ''

    for sentence in sentences:
        temp_chunk = current_chunk + ' ' + sentence.strip()
        tokenized = tokenizer(temp_chunk, return_tensors="pt", truncation=False)

        if tokenized.input_ids.shape[1] < max_length:
            current_chunk = temp_chunk.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence.strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Total de chunks: {len(chunks)}")
    return chunks

def translate_chunk(chunk):
    forced_chunk = ">>pt<< " + chunk  # força saída em português
    tokens = tokenizer(forced_chunk, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}  # mover inputs para GPU/CPU

    translated = model.generate(
        **tokens,
        max_length=512,
        num_beams=5,
        no_repeat_ngram_size=4,
        early_stopping=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

    out = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return out[0]

def postprocess_text(text):
    # Remove repetições consecutivas de palavras
    text = re.sub(r'\b(\w{3,})( \1\b)+', r'\1', text)

    # Espaços duplicados
    text = re.sub(r'\s+', ' ', text)

    # Espaços antes de pontuação
    text = re.sub(r'\s+([.,;!?])', r'\1', text)

    # Corrigir espaços depois de pontuação
    text = re.sub(r'([.,;!?])([^\s])', r'\1 \2', text)

    # Maiúsculas no início de sentença
    text = re.sub(r'([.!?]\s+)(\w)', lambda m: m.group(1) + m.group(2).upper(), text)
    text = text[0].upper() + text[1:] if text else text

    return text.strip()

def translate_text(text):
    chunks = chunk_text(text)

    results = []
    print(f"Iniciando tradução de {len(chunks)} chunks...")
    for chunk in tqdm(chunks, desc="Traduzindo"):
        result = translate_chunk(chunk)
        results.append(result)

    raw_translation = ' '.join(results)
    processed_translation = postprocess_text(raw_translation)

    return processed_translation

# ============================ #
#       BLOCO PRINCIPAL        #
# ============================ #

input_file = input("Digite o caminho do arquivo TXT para traduzir: ").strip()

if not os.path.exists(input_file):
    print("Arquivo não encontrado!")
    exit()

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("Iniciando tradução...")
translated_content = translate_text(content)

output_file = input_file.replace('.txt', '_traduzido.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(translated_content)

print(f"Tradução concluída. Arquivo salvo como {output_file}")

