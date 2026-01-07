# roberta_inference.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_tokenizer():
    """加载RoBERTa-wwm-ext-large模型和分词器"""
    print("加载RoBERTa-wwm-ext-large模型...")
    
    # 模型名称
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
    
    # 注意：使用BertTokenizer和BertModel加载，而不是RobertaTokenizer/RobertaModel
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    
    print(f"模型加载完成: {MODEL_NAME}")
    return tokenizer, model

def extract_features(texts, tokenizer, model, batch_size=8):
    """提取文本特征"""
    print("提取文本特征...")
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 分词和编码
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # 前向传播
            outputs = model(**encoded)
            
            # 取[CLS]位置的隐藏状态作为句子表示
            # shape: (batch_size, hidden_size)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_features.append(cls_embeddings.numpy())
            
            # 显示进度
            if (i // batch_size) % 10 == 0:
                print(f"  进度: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    # 合并所有batch的特征
    features = np.vstack(all_features)
    print(f"特征提取完成，形状: {features.shape}")
    return features

def simple_classifier(features, labels):
    """简单分类器（逻辑回归）"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    print("训练简单分类器...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 训练逻辑回归
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    return y_test, y_pred, clf

def plot_results(y_true, y_pred, save_dir="results"):
    """绘制评估图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'])
    plt.title('RoBERTa-wwm-ext-large 混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
    plt.show()
    
    # 2. 准确率饼图
    accuracy = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.pie([accuracy, 1-accuracy], 
            labels=[f'正确 ({accuracy:.1%})', f'错误 ({1-accuracy:.1%})'],
            colors=['#66b3ff', '#ff9999'],
            autopct='%1.1f%%')
    plt.title(f'RoBERTa-wwm-ext-large 分类准确率: {accuracy:.3f}')
    plt.savefig(f"{save_dir}/accuracy_pie.png", dpi=300)
    plt.show()
    
    return accuracy

def main():
    """主函数"""
    print("="*60)
    print("RoBERTa-wwm-ext-large 情感分析推理")
    print("="*60)
    
    start_time = time.time()
    
    # 1. 加载数据
    print("\n1. 加载数据集...")
    data_path = 'E:/SUES_course/ML_Final/dataset/LSTM_data.csv'
    data = pd.read_csv(data_path)
    
    # 取部分数据测试（避免太长）
    sample_size = min(500, len(data))  # 最多500条
    texts = data['review'].iloc[:sample_size].tolist()
    labels = data['label'].iloc[:sample_size].tolist()
    
    print(f"数据集: {len(texts)} 条评论")
    print(f"标签分布: 正面 {sum(labels)} 条, 负面 {len(labels)-sum(labels)} 条")
    
    # 2. 加载模型
    tokenizer, model = load_model_and_tokenizer()
    
    # 3. 提取特征
    features = extract_features(texts, tokenizer, model)
    
    # 4. 训练分类器并评估
    y_test, y_pred, classifier = simple_classifier(features, labels)
    
    # 5. 输出结果
    print("\n" + "="*60)
    print("评估结果:")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['负面', '正面']))
    
    # 6. 绘制图表
    print("\n生成可视化图表...")
    plot_results(y_test, y_pred)
    
    # 7. 推理示例
    print("\n" + "="*60)
    print("推理示例:")
    print("="*60)
    
    example_texts = [
        "这个手机质量很好，拍照效果很棒！",
        "物流太慢了，等了半个月才到货",
        "客服态度很差，问题一直不解决"
    ]
    
    # 提取示例特征
    example_features = extract_features(example_texts, tokenizer, model, batch_size=len(example_texts))
    example_preds = classifier.predict(example_features)
    
    for i, (text, pred) in enumerate(zip(example_texts, example_preds)):
        sentiment = "正面" if pred == 1 else "负面"
        print(f"{i+1}. {text}")
        print(f"   预测情感: {sentiment}")
        print()
    
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")
    print(f"结果已保存到 results/ 目录")
    print("="*60)

if __name__ == "__main__":
    main()