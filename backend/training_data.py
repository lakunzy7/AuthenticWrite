"""
Training Data Loader for AuthentiWrite
Loads the training dataset from CSV.
Sources:
  - aadityaubhat/GPT-wiki-intro (Wikipedia intros)
  - dmitva/human_ai_generated_text (essays)
  - Ateeqq/AI-and-Human-Generated-Text (scientific abstracts)
"""

import csv
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), 'training_dataset.csv')


def load_training_data():
    """Load training texts and labels from CSV.

    Returns:
        texts: list of str
        labels: list of int (0 = human, 1 = AI)
    """
    texts = []
    labels = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            texts.append(row[0])
            labels.append(int(row[1]))
    return texts, labels


def download_all_datasets():
    """Download and merge all 3 datasets into training_dataset.csv."""
    from datasets import load_dataset

    all_texts = []
    all_labels = []

    # Dataset 1: GPT-wiki-intro (Wikipedia intros)
    print("Downloading Dataset 1: GPT-wiki-intro (Wikipedia intros)...")
    ds = load_dataset('aadityaubhat/GPT-wiki-intro', split='train', streaming=True)
    human_count = ai_count = 0
    TARGET = 300
    for row in ds:
        wiki = row['wiki_intro'].strip()
        gen = row['generated_intro'].strip()
        if len(wiki.split()) < 50 or len(gen.split()) < 50:
            continue
        if human_count < TARGET:
            all_texts.append(wiki)
            all_labels.append(0)
            human_count += 1
        if ai_count < TARGET:
            all_texts.append(gen)
            all_labels.append(1)
            ai_count += 1
        if human_count >= TARGET and ai_count >= TARGET:
            break
    print(f"  Got {human_count} human + {ai_count} AI Wikipedia intros")

    # Dataset 2: dmitva/human_ai_generated_text (essays)
    print("Downloading Dataset 2: dmitva/human_ai_generated_text (essays)...")
    ds = load_dataset('dmitva/human_ai_generated_text', split='train', streaming=True)
    human_count = ai_count = 0
    for row in ds:
        h_text = row.get('human_text', '').strip()
        a_text = row.get('ai_text', '').strip()
        if human_count < TARGET and len(h_text.split()) >= 50:
            all_texts.append(h_text)
            all_labels.append(0)
            human_count += 1
        if ai_count < TARGET and len(a_text.split()) >= 50:
            all_texts.append(a_text)
            all_labels.append(1)
            ai_count += 1
        if human_count >= TARGET and ai_count >= TARGET:
            break
    print(f"  Got {human_count} human + {ai_count} AI essays")

    # Dataset 3: Ateeqq/AI-and-Human-Generated-Text (scientific abstracts)
    print("Downloading Dataset 3: Ateeqq/AI-and-Human-Generated-Text (scientific)...")
    ds = load_dataset('Ateeqq/AI-and-Human-Generated-Text', split='train', streaming=True)
    human_count = ai_count = 0
    for row in ds:
        text = row.get('abstract', '').strip()
        label = int(row.get('label', -1))
        if len(text.split()) < 50:
            continue
        if label == 0 and human_count < TARGET:
            all_texts.append(text)
            all_labels.append(0)
            human_count += 1
        elif label == 1 and ai_count < TARGET:
            all_texts.append(text)
            all_labels.append(1)
            ai_count += 1
        if human_count >= TARGET and ai_count >= TARGET:
            break
    print(f"  Got {human_count} human + {ai_count} AI scientific abstracts")

    # Write merged CSV
    with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for text, label in zip(all_texts, all_labels):
            writer.writerow([text, label])

    total_human = sum(1 for l in all_labels if l == 0)
    total_ai = sum(1 for l in all_labels if l == 1)
    print(f"\nSaved {len(all_texts)} samples ({total_human} human + {total_ai} AI) to {DATA_FILE}")
    return all_texts, all_labels


if __name__ == '__main__':
    import sys
    if '--download' in sys.argv:
        download_all_datasets()
    else:
        texts, labels = load_training_data()
        human = sum(1 for l in labels if l == 0)
        ai = sum(1 for l in labels if l == 1)
        print(f"Loaded {len(texts)} samples: {human} human, {ai} AI")
        print(f"Avg human length: {sum(len(t.split()) for t, l in zip(texts, labels) if l == 0) // human} words")
        print(f"Avg AI length: {sum(len(t.split()) for t, l in zip(texts, labels) if l == 1) // ai} words")
