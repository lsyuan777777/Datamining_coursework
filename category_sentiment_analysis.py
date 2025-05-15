import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BERTAnalyzer:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()

    def embed(self, text, max_length=128):
        if not isinstance(text, str) or not text.strip():
            return None
        tokens = self.tokenizer(text,
                                return_tensors='pt',
                                truncation=True,
                                padding='max_length',
                                max_length=max_length)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.model(**tokens)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().flatten()

# Statistical features extractor
def extract_statistical_features(text):
    words = nltk.word_tokenize(text)
    length = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    cap_ratio = sum(1 for c in text if c.isupper()) / length if length else 0
    excl_count = text.count('!')
    ques_count = text.count('?')
    return [length, avg_word_len, cap_ratio, excl_count, ques_count]

# Filter ratings by quantiles and analyze distribution
def prepare_data_focus(df, lower_q=0.025, upper_q=0.975, analyze_distribution=True):
    df = df[df['review_content'].apply(lambda x: isinstance(x, str))].copy()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    if analyze_distribution:
        # Analyze original distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='rating', bins=50)
        plt.title('Original Rating Distribution')
        
        # Get unique ratings and their frequencies
        rating_counts = df['rating'].value_counts().sort_index()
        print("\nOriginal Rating Distribution:")
        print(rating_counts)
        print(f"\nUnique ratings count: {len(rating_counts)}")
    
    lower = df['rating'].quantile(lower_q)
    upper = df['rating'].quantile(upper_q)
    df_filtered = df[(df['rating'] >= lower) & (df['rating'] <= upper)].reset_index(drop=True)
    
    if analyze_distribution:
        # Analyze filtered distribution
        plt.subplot(1, 2, 2)
        sns.histplot(data=df_filtered, x='rating', bins=50)
        plt.title(f'Filtered Rating Distribution ({lower_q*100:.1f}%-{upper_q*100:.1f}%)')
        plt.tight_layout()
        plt.show()
        
        # Get filtered unique ratings and their frequencies
        filtered_counts = df_filtered['rating'].value_counts().sort_index()
        print("\nFiltered Rating Distribution:")
        print(filtered_counts)
        print(f"\nFiltered unique ratings count: {len(filtered_counts)}")
    
    print(f"\nFiltering ratings to {lower_q*100:.1f}%-{upper_q*100:.1f}% interval: {lower:.2f} - {upper:.2f}")
    print(f"Samples before: {len(df)}, after: {len(df_filtered)}")
    return df_filtered

# Build combined features
def build_features_and_labels(df_filtered, bert_model, max_length=128):
    sia = SentimentIntensityAnalyzer()
    bert_feats = []
    stat_feats = []
    texts = []
    ratings = []
    for text, rating in tqdm(zip(df_filtered['review_content'], df_filtered['rating']), total=len(df_filtered), desc='Feature extraction'):
        emb = bert_model.embed(text, max_length=max_length)
        if emb is None:
            continue
        vs = sia.polarity_scores(text)
        s_feats = [vs['neg'], vs['neu'], vs['pos'], vs['compound']] + extract_statistical_features(text)
        bert_feats.append(emb)
        stat_feats.append(s_feats)
        texts.append(text)
        ratings.append(rating)
    X_bert = np.vstack(bert_feats)
    X_stat = np.array(stat_feats)
    y = np.array(ratings)
    print(f"Built BERT feats {X_bert.shape}, Stat feats {X_stat.shape}, Samples {y.shape}")
    # TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    print(f"TF-IDF+SVD features: {X_svd.shape}")
    # Combine all
    X = np.hstack([X_bert, X_stat, X_svd])
    print(f"Combined feature matrix X: {X.shape}, y: {y.shape}")
    return X, y

# Optimize with ensemble and stacking
def optimize_and_evaluate(X, y, tol=0.2, cv_splits=3):
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(200, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"After PCA: {X_pca.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Base models
    base_models = [
        ('ridge', Ridge(alpha=1)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]
    
    def evaluate_predictions(y_true, y_pred, tolerance=0.2):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        errors = np.abs(y_pred - y_true)
        tol_acc = np.mean(errors <= tolerance)
        error_stats = {
            'within_0.1': np.mean(errors <= 0.1),
            'within_0.2': np.mean(errors <= 0.2),
            'within_0.3': np.mean(errors <= 0.3),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors)
        }
        return mse, mae, tol_acc, error_stats
    
    # Stacking
    print("\nTraining Stacking Ensemble...")
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=SVR(C=1, epsilon=0.1),
        cv=cv_splits,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    preds_s = stack.predict(X_test)
    mse, mae, tol_acc, error_stats = evaluate_predictions(y_test, preds_s, tol)
    
    print(f"\nStacking Results:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"Accuracy within ±{tol:.1f}: {tol_acc:.4f}")
    print("Error Distribution:")
    for metric, value in error_stats.items():
        print(f"  {metric}: {value:.4f}")
    
    # Voting ensemble
    print("\nTraining Voting Ensemble...")
    voting = VotingRegressor(
        estimators=base_models + [('stack', stack)],
        n_jobs=-1
    )
    voting.fit(X_train, y_train)
    preds_v = voting.predict(X_test)
    mse_v, mae_v, tol_acc_v, error_stats_v = evaluate_predictions(y_test, preds_v, tol)
    
    print(f"\nVoting Results:")
    print(f"MSE: {mse_v:.4f}, MAE: {mae_v:.4f}")
    print(f"Accuracy within ±{tol:.1f}: {tol_acc_v:.4f}")
    print("Error Distribution:")
    for metric, value in error_stats_v.items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'stacking': (mse, mae, tol_acc, error_stats, preds_s, y_test),
        'voting': (mse_v, mae_v, tol_acc_v, error_stats_v, preds_v, y_test)
    }

if __name__ == '__main__':
    # Load data
    df = None
    try:
        df = pd.read_csv('amazon.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('amazon.csv', encoding='latin1')
    
    # Analyze and filter data with different quantile ranges
    print("Analyzing rating distributions with different quantile ranges...")
    quantile_ranges = [(0.025, 0.975), (0.05, 0.95), (0.1, 0.9)]
    best_range = None
    best_unique_count = float('inf')
    
    for lower_q, upper_q in quantile_ranges:
        print(f"\nTrying quantile range: {lower_q*100:.1f}%-{upper_q*100:.1f}%")
        df_focus = prepare_data_focus(df, lower_q=lower_q, upper_q=upper_q)
        unique_ratings = len(df_focus['rating'].unique())
        if 10 <= unique_ratings <= 20:  # We prefer having 10-20 unique ratings
            if best_range is None or unique_ratings < best_unique_count:
                best_range = (lower_q, upper_q)
                best_unique_count = unique_ratings
    
    if best_range is None:
        print("\nNo ideal range found, using default 95% range")
        best_range = (0.025, 0.975)
    
    print(f"\nSelected best range: {best_range[0]*100:.1f}%-{best_range[1]*100:.1f}%")
    df_focus = prepare_data_focus(df, lower_q=best_range[0], upper_q=best_range[1])
    
    # Continue with BERT and NLTK analysis
    print("\nInitializing BERT analyzer...")
    bert_analyzer = BERTAnalyzer()
    print("\nBuilding features...")
    X, y = build_features_and_labels(df_focus, bert_analyzer)
    print("\nOptimizing and evaluating models...")
    results = optimize_and_evaluate(X, y)
    
    print("\nFinal Results:")
    for name, (mse, mae, tol_acc, error_stats, preds, y_true) in results.items():
        print(f"{name}: MSE={mse:.4f}, MAE={mae:.4f}, TolAcc={tol_acc:.4f}")
        print("Error Distribution:")
        for metric, value in error_stats.items():
            print(f"  {metric}: {value:.4f}")
