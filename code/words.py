import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 数据预处理函数
def preprocess_text(raw_text):
    # 1. 移除HTML标签
    text = BeautifulSoup(raw_text, features="html.parser").get_text()
    
    # 2. 特殊处理标点符号，保留情感信息
    text = text.replace('!', ' EXCLAMATION ')
    text = text.replace('?', ' QUESTION ')
    
    # 3. 移除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    
    # 4. 转换为小写并分词
    words = letters_only.lower().split()
    
    # 5. 处理特殊标记
    processed_words = []
    for word in words:
        if word == 'exclamation':
            processed_words.append('!')
        elif word == 'question':
            processed_words.append('?')
        else:
            processed_words.append(word)
    
    # 6. 移除停用词
    stops = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now'
    }
    
    meaningful_words = [w for w in processed_words if not w in stops]
    return " ".join(meaningful_words)

def main():
    print("=" * 60)
    print("情感分析模型 - 基于TF-IDF和逻辑回归")
    print("=" * 60)
    
    # 1. 读取训练数据
    print("\n1. 读取训练数据...")
    train_data = pd.read_csv("data/labeledTrainData.tsv", delimiter="\t", quoting=3)
    print(f"   训练数据行数: {len(train_data)}")
    
    # 2. 预处理训练数据
    print("\n2. 预处理训练数据...")
    clean_train_reviews = []
    total_reviews = len(train_data)
    
    for i, review in enumerate(train_data["review"]):
        if (i + 1) % 1000 == 0:
            print(f"   已处理 {i + 1} / {total_reviews} 条评论")
        clean_train_reviews.append(preprocess_text(review))
    
    # 3. 创建TF-IDF特征
    print("\n3. 创建TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    train_features = tfidf_vectorizer.fit_transform(clean_train_reviews)
    print(f"   特征矩阵形状: {train_features.shape}")
    
    # 4. 训练逻辑回归模型
    print("\n4. 训练逻辑回归模型...")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(train_features, train_data["sentiment"])
    print("   模型训练完成")
    
    # 6. 读取测试数据
    print("\n6. 读取测试数据...")
    test_data = pd.read_csv("data/testData.tsv", delimiter="\t", quoting=3)
    print(f"   测试数据行数: {len(test_data)}")
    
    # 6. 预处理测试数据
    print("\n6. 预处理测试数据...")
    clean_test_reviews = []
    total_test_reviews = len(test_data)
    
    for i, review in enumerate(test_data["review"]):
        if (i + 1) % 1000 == 0:
            print(f"   已处理 {i + 1} / {total_test_reviews} 条评论")
        clean_test_reviews.append(preprocess_text(review))
    
    # 7. 提取测试数据特征
    print("\n7. 提取测试数据特征...")
    test_features = tfidf_vectorizer.transform(clean_test_reviews)
    
    # 8. 预测测试数据
    print("\n8. 预测测试数据...")
    # 预测概率（适用于ROC AUC评估）
    predicted_probabilities = model.predict_proba(test_features)[:, 1]
    
    # 9. 生成提交文件
    print("\n9. 生成提交文件...")
    
    # 生成包含预测概率的提交文件
    submission = pd.DataFrame({
        "id": test_data["id"],
        "sentiment": predicted_probabilities
    })
    submission.to_csv("results/Submission.csv", index=False, quoting=3)
    
    print("\n" + "=" * 60)
    print("任务完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("results/Submission.csv - 包含情感分析预测概率")
    print("\n文件格式符合要求，可直接提交。")

if __name__ == "__main__":
    main()
