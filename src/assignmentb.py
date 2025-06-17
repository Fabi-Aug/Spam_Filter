# Data Science I - Spam Filter Assignment
# Student: [Your Name]
# Date: June 2025
# Assignment: Building a spam classifier using Apache SpamAssassin dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import re
import os
from pathlib import Path

# ## 1. Data Loading and Preprocessing

def load_emails_from_directory(directory_path, label):
    """
    Load emails from a directory and assign labels
    
    Args:
        directory_path: Path to directory containing email files
        label: 0 for ham, 1 for spam
    
    Returns:
        List of tuples (email_content, label)
    """
    emails = []
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    emails.append((content, label))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return emails

# ## 2. Data Preparation Pipeline

class EmailPreprocessor:
    """
    Email preprocessing pipeline with configurable hyperparameters
    """
    
    def __init__(self, 
                 strip_headers=True,
                 to_lowercase=True,
                 remove_punctuation=True,
                 replace_urls=True,
                 replace_numbers=True):
        """
        Initialize preprocessor with hyperparameters
        
        Args:
            strip_headers: Remove email headers
            to_lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            replace_urls: Replace URLs with 'URL'
            replace_numbers: Replace numbers with 'NUMBER'
        """
        self.strip_headers = strip_headers
        self.to_lowercase = to_lowercase
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
    
    def preprocess_email(self, email_content):
        """
        Preprocess a single email
        
        Args:
            email_content: Raw email content
            
        Returns:
            Preprocessed email text
        """
        text = email_content
        
        # Strip email headers (everything before first blank line)
        if self.strip_headers:
            lines = text.split('\n')
            body_start = 0
            for i, line in enumerate(lines):
                if line.strip() == '':
                    body_start = i + 1
                    break
            text = '\n'.join(lines[body_start:])
        
        # Convert to lowercase
        if self.to_lowercase:
            text = text.lower()
        
        # Replace URLs with 'URL'
        if self.replace_urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text = re.sub(url_pattern, 'URL', text)
        
        # Replace numbers with 'NUMBER'
        if self.replace_numbers:
            text = re.sub(r'\d+', 'NUMBER', text)
        
        # Remove punctuation (keep only letters, numbers, spaces)
        if self.remove_punctuation:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text

# ## 3. Load Data from Folders
# Load actual SpamAssassin dataset from data folder

def load_dataset():
    """
    Load emails from easy_ham_2 and spam_2 folders
    
    Returns:
        Tuple of (emails_list, labels_list)
    """
    # Define data paths
    data_folder = "data"
    ham_folder = os.path.join(data_folder, "easy_ham_2")
    spam_folder = os.path.join(data_folder, "spam_2")
    
    print(f"Loading ham emails from: {ham_folder}")
    print(f"Loading spam emails from: {spam_folder}")
    
    # Load ham and spam emails
    ham_emails = load_emails_from_directory(ham_folder, 0)  # 0 = ham
    spam_emails = load_emails_from_directory(spam_folder, 1)  # 1 = spam
    
    print(f"Loaded {len(ham_emails)} ham emails")
    print(f"Loaded {len(spam_emails)} spam emails")
    
    # Combine all emails
    all_emails = ham_emails + spam_emails
    
    # Shuffle the data
    np.random.seed(42)
    np.random.shuffle(all_emails)
    
    # Separate features and labels
    emails = [email[0] for email in all_emails]
    labels = [email[1] for email in all_emails]
    
    return emails, labels

# ## 4. Model Training and Evaluation

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classifiers and evaluate their performance
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        
    Returns:
        Dictionary of trained models and their performance metrics
    """
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, kernel='linear')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred
        }
    
    return results

# ## 5. Main Execution

def main():
    """
    Main function to execute the spam filter pipeline
    """
    print("Spam Filter Assignment - Data Science I")
    print("=" * 50)
    
    # 1) Load from your SpamAssassin folders under data/
    print("Loading email data from disk...")
    emails, labels = load_dataset()
    print(f"Total emails: {len(emails)}")
    print(f"Ham emails:  {labels.count(0)}")
    print(f"Spam emails: {labels.count(1)}")

    
    # Initialize preprocessor with hyperparameters
    preprocessor = EmailPreprocessor(
        strip_headers=True,
        to_lowercase=True,
        remove_punctuation=True,
        replace_urls=True,
        replace_numbers=True
    )
    
    # Preprocess emails
    print("\nPreprocessing emails...")
    processed_emails = [preprocessor.preprocess_email(email) for email in emails]
    
    # Create feature vectors using CountVectorizer (bag of words)
    print("Creating feature vectors...")
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(processed_emails)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Find best model based on combined precision and recall (F1-like)
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['precision'] + results[x]['recall'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Precision: {results[best_model_name]['precision']:.3f}")
    print(f"Recall: {results[best_model_name]['recall']:.3f}")
    
    # Demonstrate the spam filter with a new email
    print("\n" + "="*50)
    print("SPAM FILTER DEMONSTRATION")
    print("="*50)
    
    test_email = """
    Subject: Congratulations! You've won $1,000,000!!!
    
    Dear lucky winner,
    
    You have been selected to receive ONE MILLION DOLLARS absolutely FREE!
    No purchase necessary! Act now before this amazing offer expires!
    
    Click here to claim your prize: http://fake-lottery-site.com
    
    Congratulations again!
    """
    
    # Preprocess and predict
    processed_test = preprocessor.preprocess_email(test_email)
    test_vector = vectorizer.transform([processed_test])
    prediction = best_model.predict(test_vector)[0]
    probability = best_model.predict_proba(test_vector)[0] if hasattr(best_model, 'predict_proba') else None
    
    print("Test Email:")
    print(test_email[:200] + "...")
    print(f"\nPrediction: {'SPAM' if prediction == 1 else 'HAM'}")
    if probability is not None:
        print(f"Confidence: Ham={probability[0]:.3f}, Spam={probability[1]:.3f}")
    
    return best_model, vectorizer, preprocessor

# ## 6. Hyperparameter Experimentation

def experiment_with_hyperparameters():
    """
    Experiment with different preprocessing hyperparameters
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER EXPERIMENTATION")
    print("="*50)
    
    # Different preprocessing configurations
    configs = [
        {'strip_headers': True, 'to_lowercase': True, 'remove_punctuation': True, 'replace_urls': True, 'replace_numbers': True},
        {'strip_headers': False, 'to_lowercase': True, 'remove_punctuation': True, 'replace_urls': True, 'replace_numbers': True},
        {'strip_headers': True, 'to_lowercase': False, 'remove_punctuation': True, 'replace_urls': True, 'replace_numbers': True},
        {'strip_headers': True, 'to_lowercase': True, 'remove_punctuation': False, 'replace_urls': True, 'replace_numbers': True},
    ]
    
    # load the same data as in main()
    emails, labels = load_dataset()
    
    best_config = None
    best_score = 0
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        # Preprocess with current configuration
        preprocessor = EmailPreprocessor(**config)
        processed_emails = [preprocessor.preprocess_email(email) for email in emails]
        
        # Vectorize
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(processed_emails)
        y = np.array(labels)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Use Naive Bayes for quick evaluation
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        combined_score = precision + recall
        
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, Combined: {combined_score:.3f}")
        
        if combined_score > best_score:
            best_score = combined_score
            best_config = config
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best combined score: {best_score:.3f}")

# Run the main function
if __name__ == "__main__":
    model, vectorizer, preprocessor = main()
    experiment_with_hyperparameters()

