import os


def align_and_clean(source_path, target_path, clean_source_path, clean_target_path):
    # Ler os arquivos originais
    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f if line.strip()]

    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f if line.strip()]

    # Verificar se o número de linhas é igual
    if len(source_lines) != len(target_lines):
        print(f"⚠️ Atenção: Número de linhas diferente.")
        print(f"Source: {len(source_lines)} linhas")
        print(f"Target: {len(target_lines)} linhas")

        min_len = min(len(source_lines), len(target_lines))
        source_lines = source_lines[:min_len]
        target_lines = target_lines[:min_len]
        print(f"⚠️ Linhas foram cortadas para alinhar: {min_len} linhas")

    # Garantir que a pasta clean exista
    clean_dir = os.path.dirname(clean_source_path)
    os.makedirs(clean_dir, exist_ok=True)

    # Salvar arquivos alinhados na pasta clean
    with open(clean_source_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(source_lines) + "\n")

    with open(clean_target_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(target_lines) + "\n")

    print(f"✅ Arquivos alinhados salvos em:\n  {clean_source_path}\n  {clean_target_path}")


if __name__ == "__main__":
    base_dir = os.path.join('..', 'data')
    raw_dir = os.path.join(base_dir, 'raw')
    clean_dir = os.path.join(base_dir, 'clean')

    source_file = os.path.join(raw_dir, 'source.txt')
    target_file = os.path.join(raw_dir, 'target.txt')

    clean_source_file = os.path.join(clean_dir, 'source.txt')
    clean_target_file = os.path.join(clean_dir, 'target.txt')

    align_and_clean(source_file, target_file, clean_source_file, clean_target_file)
