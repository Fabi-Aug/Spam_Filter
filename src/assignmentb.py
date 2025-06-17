# -*- coding: utf-8 -*-
"""
Spam Filter Assignment
Student: Fabian Augschöll
Date: June 2025
"""

import os
import re
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score


def load_emails(directory, label):
    emails = []
    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                emails.append((f.read(), label))
        except Exception:
            continue
    return emails


class EmailPreprocessor:
    def __init__(self, strip_headers=True, lowercase=True,
                 remove_punct=True, replace_urls=True, replace_nums=True):
        self.strip_headers = strip_headers
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.replace_urls = replace_urls
        self.replace_nums = replace_nums

    def preprocess(self, text):
        if self.strip_headers:
            parts = text.split('\n\n', 1)
            text = parts[1] if len(parts) > 1 else parts[0]
        if self.lowercase:
            text = text.lower()
        if self.replace_urls:
            text = re.sub(r'http[s]?://\S+', 'URL', text)
        if self.replace_nums:
            text = re.sub(r'\d+', 'NUMBER', text)
        if self.remove_punct:
            text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())


def prepare_data(base_path):
    ham_dir = Path(base_path) / 'easy_ham_2'
    spam_dir = Path(base_path) / 'spam_2'
    ham = load_emails(str(ham_dir), 0)
    spam = load_emails(str(spam_dir), 1)
    data = ham + spam
    np.random.seed(42)
    np.random.shuffle(data)
    texts, labels = zip(*data)
    return list(texts), np.array(labels)


def train_and_report(X_train, X_test, y_train, y_test):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print(f"--- {name} ---")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(classification_report(y_test, y_pred, target_names=['Ham','Spam']))
        results[name] = {'model': model, 'precision': precision, 'recall': recall}
    return results


def hyperparameter_experiment(texts, labels):
    configs = [
        {'strip_headers': True, 'lowercase': True, 'remove_punct': True, 'replace_urls': True, 'replace_nums': True},
        {'strip_headers': False, 'lowercase': True, 'remove_punct': True, 'replace_urls': True, 'replace_nums': True},
        {'strip_headers': True, 'lowercase': False, 'remove_punct': True, 'replace_urls': True, 'replace_nums': True},
        {'strip_headers': True, 'lowercase': True, 'remove_punct': False, 'replace_urls': True, 'replace_nums': True},
    ]
    best_score, best_conf = 0, None
    for i, cfg in enumerate(configs, 1):
        pre = EmailPreprocessor(**cfg)
        proc = [pre.preprocess(t) for t in texts]
        vec = CountVectorizer(max_features=1000, stop_words='english').fit_transform(proc)
        X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.3,
                                                            random_state=42, stratify=labels)
        model = MultinomialNB().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        print(f"Config {i}: P={p:.3f}, R={r:.3f}, Combined={p+r:.3f}")
        if p + r > best_score:
            best_score, best_conf = p + r, cfg
    print(f"Best config: {best_conf}, Score: {best_score:.3f}")


def main(data_path='data'):
    texts, labels = prepare_data(data_path)
    preprocessor = EmailPreprocessor()
    processed = [preprocessor.preprocess(t) for t in texts]
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(processed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels)

    print("Training and evaluation:")
    results = train_and_report(X_train, X_test, y_train, y_test)
    best = max(results, key=lambda k: results[k]['precision'] + results[k]['recall'])
    print(f"Best model: {best}\n")

    print("Demonstration:")
    sample = "Subject: Win money now! Click http://spam.link"
    proc = preprocessor.preprocess(sample)
    vec = vectorizer.transform([proc])
    pred = results[best]['model'].predict(vec)[0]
    prob = results[best]['model'].predict_proba(vec)[0]
    print(f"Sample prediction: {'SPAM' if pred else 'HAM'} (Ham={prob[0]:.2f}, Spam={prob[1]:.2f})\n")

    print("Hyperparameter experimentation:")
    hyperparameter_experiment(texts, labels)

if __name__ == '__main__':
    main()
