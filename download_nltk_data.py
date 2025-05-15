import nltk
import ssl
import os
import sys

def setup_ssl_context():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        print("Warning: Unable to create unverified HTTPS context")
        pass
    else:
        print("Setting up unverified HTTPS context")
        ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    # 设置NLTK数据目录
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    print(f"NLTK数据将被下载到: {nltk_data_dir}")
    
    # 下载必要的NLTK数据
    datasets = ['punkt', 'stopwords', 'vader_lexicon']
    for dataset in datasets:
        print(f"\n正在下载 {dataset}...")
        try:
            nltk.download(dataset, quiet=False)
            print(f"成功下载 {dataset}")
        except Exception as e:
            print(f"下载 {dataset} 时出错: {str(e)}")
            print("尝试使用备用方法下载...")
            try:
                nltk.download(dataset, download_dir=nltk_data_dir)
                print(f"使用备用方法成功下载 {dataset}")
            except Exception as e:
                print(f"备用下载方法也失败: {str(e)}")

if __name__ == "__main__":
    print("开始设置NLTK数据下载...")
    setup_ssl_context()
    download_nltk_data()
    print("\n下载过程完成!") 