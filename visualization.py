import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def plot_rating_distribution(df, save_path='plots/rating_distribution.png'):
    """Plot rating distribution"""
    # Convert rating to numeric type
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    plt.figure(figsize=(12, 6))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='rating', bins=50)
    plt.title('Original Rating Distribution', fontsize=12)
    plt.xlabel('Rating', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    
    # Calculate quantile range
    lower_q, upper_q = df['rating'].quantile([0.025, 0.975])
    df_filtered = df[(df['rating'] >= lower_q) & (df['rating'] <= upper_q)]
    
    # Filtered distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_filtered, x='rating', bins=50)
    plt.title('Filtered Rating Distribution (2.5%-97.5%)', fontsize=12)
    plt.xlabel('Rating', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_filtered

def plot_error_distribution(y_true, y_pred, save_path='plots/error_distribution.png'):
    """Plot prediction error distribution"""
    errors = np.abs(y_pred - y_true)
    
    plt.figure(figsize=(12, 6))
    
    # Error distribution histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=errors, bins=50)
    plt.axvline(x=0.1, color='r', linestyle='--', label='±0.1 Error')
    plt.axvline(x=0.2, color='g', linestyle='--', label='±0.2 Error')
    plt.axvline(x=0.3, color='b', linestyle='--', label='±0.3 Error')
    plt.title('Prediction Error Distribution', fontsize=12)
    plt.xlabel('Absolute Error', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=9)
    
    # True vs Predicted scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal Prediction')
    plt.title('True vs Predicted Values', fontsize=12)
    plt.xlabel('True Rating', fontsize=10)
    plt.ylabel('Predicted Rating', fontsize=10)
    plt.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results, save_path='plots/model_comparison.png'):
    """Plot model performance comparison"""
    plt.figure(figsize=(15, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    models = list(results.keys())
    metrics = ['within_0.1', 'within_0.2', 'within_0.3']
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[model][3][metric] for model in models]
        plt.bar(x + i*width, values, width, label=f'±{metric[-3:]} Error')
    
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Accuracy Comparison at Different Error Ranges', fontsize=12)
    plt.xticks(x + width, models, fontsize=9)
    plt.legend(fontsize=9)
    
    # MSE and MAE comparison
    plt.subplot(1, 2, 2)
    mse_values = [results[model][0] for model in models]
    mae_values = [results[model][1] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mse_values, width, label='MSE')
    plt.bar(x + width/2, mae_values, width, label='MAE')
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Error Value', fontsize=10)
    plt.title('MSE and MAE Comparison', fontsize=12)
    plt.xticks(x, models, fontsize=9)
    plt.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_report(results, save_path='plots/performance_report.png'):
    """Create comprehensive performance report"""
    plt.figure(figsize=(15, 10))
    
    # Performance metrics radar chart
    plt.subplot(2, 2, 1, projection='polar')
    models = list(results.keys())
    metrics = ['MSE', 'MAE', '±0.1 Accuracy', '±0.2 Accuracy', '±0.3 Accuracy']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    for model in models:
        values = [
            results[model][0],  # MSE
            results[model][1],  # MAE
            results[model][3]['within_0.1'],
            results[model][3]['within_0.2'],
            results[model][3]['within_0.3']
        ]
        values = [(v - min(values))/(max(values) - min(values)) for v in values]
        values += values[:1]
        angles_plot = np.concatenate((angles, [angles[0]]))
        plt.plot(angles_plot, values, 'o-', label=model)
        plt.fill(angles_plot, values, alpha=0.25)
    
    plt.xticks(angles, metrics, fontsize=9)
    plt.title('Model Performance Radar Chart', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    import os
    from category_sentiment_analysis import prepare_data_focus, build_features_and_labels, optimize_and_evaluate, BERTAnalyzer
    
    # Create directory for saving plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    try:
        # Read data
        print("Reading data...")
        df = pd.read_csv('amazon.csv')
        
        # Plot rating distribution and get filtered data
        print("Generating rating distribution plot...")
        df_filtered = plot_rating_distribution(df)
        print("Rating distribution plot saved: plots/rating_distribution.png")
        
        # Prepare data
        print("Preparing data...")
        df_focus = prepare_data_focus(df_filtered)
        
        # Initialize BERT analyzer
        print("Initializing BERT model...")
        bert_analyzer = BERTAnalyzer()
        
        # Build features and labels
        print("Building features...")
        X, y = build_features_and_labels(df_focus, bert_analyzer)
        
        # Run models and get results
        print("Training and evaluating models...")
        results = optimize_and_evaluate(X, y)
        
        # Generate model comparison plot
        print("Generating model comparison plot...")
        plot_model_comparison(results)
        print("Model comparison plot saved: plots/model_comparison.png")
        
        # Get best model predictions
        best_model = 'voting' if results['voting'][1] < results['stacking'][1] else 'stacking'
        y_pred = results[best_model][4] if len(results[best_model]) > 4 else None
        
        if y_pred is not None:
            # Generate error distribution plot
            print("Generating error distribution plot...")
            plot_error_distribution(y, y_pred)
            print("Error distribution plot saved: plots/error_distribution.png")
        
        # Generate performance report
        print("Generating performance report...")
        create_performance_report(results)
        print("Performance report saved: plots/performance_report.png")
        
        print("\nVisualization complete! All plots saved in plots directory.")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please ensure data file exists and is in correct format.")

if __name__ == "__main__":
    main() 