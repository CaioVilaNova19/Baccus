from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# Nome do modelo base
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"

# Carregar tokenizer e modelo
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Verificar se há GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Carregar dados
source_file = "./data/source.txt"
target_file = "./data/target.txt"

with open(source_file, "r", encoding="utf-8") as f:
    src_lines = [line.strip() for line in f.readlines()]

with open(target_file, "r", encoding="utf-8") as f:
    tgt_lines = [line.strip() for line in f.readlines()]

# Verificar alinhamento dos dados
assert len(src_lines) == len(tgt_lines), "O número de linhas não bate entre source e target!"

# Criar dataframe
data = pd.DataFrame({"translation_source": src_lines, "translation_target": tgt_lines})

# Criar dataset Hugging Face
dataset = Dataset.from_pandas(data)

# Função de tokenização
def preprocess_function(examples):
    inputs = [">>pt<< " + ex for ex in examples["translation_source"]]
    targets = examples["translation_target"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Aplicar a tokenização
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Dividir em treino e validação
split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# Argumentos de treinamento
training_args = Seq2SeqTrainingArguments(
    output_dir="../results_finetune",
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    predict_with_generate=True,
    fp16=True if device == "cuda" else False,
    logging_dir="../logs",
)

# Configurar o Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Iniciar o fine-tuning
trainer.train()

# Salvar o modelo
trainer.save_model("../Baccus_en_pt")
tokenizer.save_pretrained("../Baccus_en_pt")

print("Modelo fine-tunado salvo em ../Baccus_en_pt")

