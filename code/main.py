from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

app = FastAPI(title="情感分析API", description="基于TF-IDF和逻辑回归的情感分析服务")

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

# 全局变量存储模型和向量化器
model = None
vectorizer = None

# 请求模型
class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: List[str]

class PredictionResponse(BaseModel):
    review: str
    sentiment: str
    probability: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, vectorizer
    
    # 检查是否有保存的模型
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("模型加载完成")
    else:
        print("未找到保存的模型，需要在首次使用时训练")
        model = None
        vectorizer = None

@app.get("/")
async def root():
    return {"message": "情感分析API服务正在运行", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    """单个评论的情感分析"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="模型尚未训练，请先训练模型")
    
    # 预处理文本
    processed_text = preprocess_text(request.review)
    
    # 特征提取
    features = vectorizer.transform([processed_text])
    
    # 预测
    probability = model.predict_proba(features)[0, 1]
    sentiment = "积极" if probability > 0.5 else "消极"
    
    return PredictionResponse(
        review=request.review,
        sentiment=sentiment,
        probability=float(probability)
    )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchReviewRequest):
    """批量评论的情感分析"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="模型尚未训练，请先训练模型")
    
    results = []
    for review in request.reviews:
        # 预处理文本
        processed_text = preprocess_text(review)
        
        # 特征提取
        features = vectorizer.transform([processed_text])
        
        # 预测
        probability = model.predict_proba(features)[0, 1]
        sentiment = "积极" if probability > 0.5 else "消极"
        
        results.append(PredictionResponse(
            review=review,
            sentiment=sentiment,
            probability=float(probability)
        ))
    
    return BatchPredictionResponse(results=results)

@app.post("/train")
async def train_model():
    """训练模型"""
    global model, vectorizer
    
    try:
        # 读取训练数据
        train_data = pd.read_csv("data/labeledTrainData.tsv", delimiter="\t", quoting=3)
        
        # 预处理训练数据
        clean_train_reviews = []
        for review in train_data["review"]:
            clean_train_reviews.append(preprocess_text(review))
        
        # 创建TF-IDF特征
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        train_features = vectorizer.fit_transform(clean_train_reviews)
        
        # 训练逻辑回归模型
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(train_features, train_data["sentiment"])
        
        # 保存模型
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        return {"message": "模型训练完成", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
