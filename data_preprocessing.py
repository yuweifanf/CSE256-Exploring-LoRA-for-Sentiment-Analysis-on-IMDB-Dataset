import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from wordcloud import WordCloud
import spacy
import matplotlib.pyplot as plt


def extract_adjectives(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    exset = ['good', 'great', 'new']
    adjectives = [token.text for token in doc if token.pos_ == 'ADJ' and token.text.lower() not in exset]
    return ' '.join(adjectives)

def load_dataset_into_to_dataframe():
    basepath = "aclImdb"
    labels = {"pos": 1, "neg": 0}
    df = pd.DataFrame()
    with tqdm(total=50000) as pbar:
        for s in ("test", "train"):
            for l in ("pos", "neg"):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                        txt = infile.read()
                        x = pd.DataFrame([[txt, labels[l]]], columns=["review", "sentiment"])
                        df = pd.concat([df, x], ignore_index=False)
                    pbar.update()
    df.columns = ["text", "label"]
    df = df.sample(frac=1, random_state=23).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df_shuffled = load_dataset_into_to_dataframe()

    # Split and save training set
    df_train = df_shuffled.iloc[:35_000]
    df_train.to_csv(os.path.join("data", "train.csv"), index=False, encoding="utf-8")
    # Visualize the distribution
    class_distribution = np.bincount(df_train["label"].values)
    labels = [f"Class {i}" for i in range(len(class_distribution))]
    plt.bar(labels, class_distribution, tick_label=labels)
    plt.title("Training Set Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.show()

    # Split and save validation set
    df_val = df_shuffled.iloc[35_000:40_000]
    df_val.to_csv(os.path.join("data", "val.csv"), index=False, encoding="utf-8")
    # Visualize the distribution
    class_distribution = np.bincount(df_val["label"].values)
    labels = [f"Class {i}" for i in range(len(class_distribution))]
    plt.bar(labels, class_distribution, tick_label=labels)
    plt.title("Validation Set Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.show()
    
    # Split and save test set
    df_test = df_shuffled.iloc[40_000:]
    df_test.to_csv(os.path.join("data", "test.csv"), index=False, encoding="utf-8")
    # Visualize the distribution
    class_distribution = np.bincount(df_test["label"].values)
    labels = [f"Class {i}" for i in range(len(class_distribution))]
    plt.bar(labels, class_distribution, tick_label=labels)
    plt.title("Test Set Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.show()

    # Visualize adjectives wordcloud of all sets
    train_file_path = 'data/train.csv'
    validation_file_path = 'data/val.csv'
    test_file_path = 'data/test.csv'
    font_path = 'arial.ttf'

    train_texts = []
    with open(train_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            index, text, label = parts
            train_texts.append(text)
    train_texts_combined = ' '.join([extract_adjectives(t) for t in train_texts])
    train_wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path, max_words=20).generate(train_texts_combined)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(train_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud of training set (Adjectives Only)')
    plt.show()

    validation_texts = []
    with open(validation_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            index, text, label = parts
            validation_texts.append(text)
    validation_texts_combined = ' '.join([extract_adjectives(t) for t in validation_texts])
    validation_wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path, max_words=20).generate(train_texts_combined)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(validation_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud of validation set (Adjectives Only)')
    plt.show()

    test_texts = []
    with open(test_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            index, text, label = parts
            test_texts.append(text)
    test_texts_combined = ' '.join([extract_adjectives(t) for t in test_texts])
    test_wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path, max_words=20).generate(train_texts_combined)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(test_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud of test set (Adjectives Only)')
    plt.show()


