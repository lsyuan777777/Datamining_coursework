# E-commerce Review Sentiment Analysis System

## Project Overview

This project implements a machine learning-based review text sentiment score prediction system that automatically predicts rating values through text content analysis. The system combines BERT text embedding, NLTK sentiment analysis, and various machine learning models to achieve high prediction accuracy. Data concentration analysis ensures the quality and representativeness of training data.

## Project Structure

```
Z_DATAMINING_WORK/
├── plots/                    # Data visualization output directory
│   ├── rating_distribution.png    # Rating distribution plot
│   ├── model_comparison.png       # Model comparison plot
├── amazon.csv               # Original dataset (Amazon e-commerce platform review data)
├── category_sentiment_analysis.py  # Core sentiment analysis implementation
├── download_nltk_data.py    # NLTK data download and initialization script
├── visualization.py         # Data visualization implementation module
├── requirements.txt        # Project dependency package list
└── README.md              # Project description document
```

### Directory Details

1. **`plots/`**
2. - Output directory for visualization results
   - Created automatically when running analysis
   - Can be safely deleted; will be recreated when generating new plots

## Technical Features

1. **Text Feature Extraction**

   - BERT pre-trained model for text embedding
   - NLTK for sentiment analysis
   - Statistical feature extraction
   - TF-IDF vectorization
2. **Model Integration**

   - Stacking ensemble learning
   - Voting ensemble learning
   - Multiple base models: Ridge, Random Forest, Gradient Boosting
3. **Feature Engineering**

   - PCA dimensionality reduction
   - Standardization processing
   - Text statistical features

## Performance Metrics

Main performance metrics on the test set:

### Stacking Model Performance

- MSE: 0.0316
- MAE: 0.1364
- Prediction Accuracy:
  * Within ±0.1 range: 47.81%
  * Within ±0.2 range: 75.91%
  * Within ±0.3 range: 88.69%
- Average Error: 0.1364
- Median Error: 0.1028

### Voting Model Performance

- MSE: 0.0305
- MAE: 0.1351
- Prediction Accuracy:
  * Within ±0.1 range: 50.00%
  * Within ±0.2 range: 74.82%
  * Within ±0.3 range: 90.15%
- Average Error: 0.1351
- Median Error: 0.0999

## Environment Requirements

```
python >= 3.6
torch
transformers
nltk
scikit-learn
pandas
numpy
tqdm
matplotlib
seaborn
```

## Installation and Setup

### Method 1: Using Virtual Environment (Recommended)

1. Install Python virtual environment if not installed:

```bash
pip install virtualenv
```

2. Create and activate a new virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For Windows:
.venv\Scripts\activate
# For Linux/Mac:
source .venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

### Method 2: Direct Installation

If you don't want to use a virtual environment:

1. Install required packages globally:

```bash
pip install torch transformers nltk scikit-learn pandas numpy tqdm matplotlib seaborn
```

2. Download NLTK data:

```bash
python download_nltk_data.py
```

### Verify Installation

To verify the installation:

```bash
# Check Python version
python --version  # Should be >= 3.6

# Check package installation
python -c "import torch; import transformers; import nltk; print('All packages installed successfully!')"
```

### Troubleshooting

If you encounter any issues:

1. Package installation errors:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --no-cache-dir
   ```
2. CUDA/GPU issues:

   ```bash
   # Check CUDA availability
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
3. Memory issues:

   - Reduce batch size in code
   - Use CPU mode if GPU memory is insufficient

## Usage Instructions

1. **Data Preparation**

   - Ensure the data file (amazon.csv) contains:
     * review_content: Review text
     * rating: Rating value
2. **Data Processing**

```python
# Run the main analysis
python category_sentiment_analysis.py
```

3. **View Results**
   - Check visualization results in the `plots/` directory
   - Analysis results will be printed to console

## Implementation Details

### Data Processing

- Data filtering and cleaning
- Feature extraction using BERT and NLTK
- Statistical feature calculation

### Model Training

- Multiple base model training
- Ensemble learning implementation
- Cross-validation and evaluation

### Visualization

- Rating distribution analysis
- Error distribution visualization
- Model performance comparison

## Results Analysis

The system achieves:

- High accuracy within ±0.2 error range (>75%)
- Stable performance across different review types
- Effective feature extraction and combination
- Reliable sentiment prediction capability

## Future Improvements

1. **Model Optimization**

   - Fine-tuning of BERT parameters
   - Exploration of new ensemble methods
   - Performance optimization
2. **Feature Enhancement**

   - Additional text features
   - Context consideration
   - User behavior integration

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Contact

For any questions or suggestions, please open an issue in the repository.

## Acknowledgments

- BERT model from Google Research
- NLTK development team
- Scikit-learn community
