import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
import pickle
import os
import warnings
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子，保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 忽略警告
warnings.filterwarnings('ignore')


# 数据加载与预处理
def load_and_preprocess_data(file_path, stopwords_path):
    """
    加载数据并进行预处理
    """
    # 读取数据
    data = pd.read_csv(file_path)
    texts = data['review'].tolist()
    labels = data['label'].tolist()

    # 检查数据加载情况
    print("数据加载完成，共有 {} 条数据".format(len(texts)))
    print("前5条数据文本：")
    for i in range(5):
        print(texts[i][:50], "...")
    print("\n前5条数据标签：")
    print(labels[:5])

    # 加载停用词表
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"停用词表文件 {stopwords_path} 不存在，请检查路径")

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set()
        for line in f:
            word = line.strip()
            # 清理各种空白字符和特殊空格
            word = word.replace('\u3000', '').replace('\xa0', '').strip()
            if word:  # 只添加非空词
                stopwords.add(word)

    # 添加常用的中文标点符号和数字作为停用词（确保过滤有效）
    additional_stopwords = {
        '，', '。', '！', '？', '；', '：', '「', '」', '（', '）', '【', '】', '『', '』',
        ',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上'
    }
    stopwords.update(additional_stopwords)

    print("\n停用词表加载完成，共有 {} 个停用词".format(len(stopwords)))
    print("部分停用词：")
    # 显示前20个，确保能看到实际内容
    for i, word in enumerate(list(stopwords)[:20]):
        print(f"  {i+1:2d}. '{word}'")

    # 文本预处理：分词和去停用词
    def preprocess_text(text):
        words = jieba.lcut(text)
        return ' '.join([word for word in words if word not in stopwords])

    processed_texts = [preprocess_text(text) for text in texts]

    print("\n文本预处理完成，前5条处理后的文本：")
    for i in range(5):
        print(processed_texts[i][:50], "...")  # 打印前50个字符

    return processed_texts, labels


# 文本向量化
def text_vectorization(texts, max_len):
    """
    对文本进行向量化处理
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    # 填充序列
    data = pad_sequences(sequences, maxlen=max_len)

    print("\n文本向量化完成")
    print("词汇表大小：{}".format(len(word_index) + 1))
    print("前5条向量化后的序列：")
    print(data[:5])

    return data, tokenizer, word_index


# 模型构建
def build_lstm_model(vocab_size, embedding_dim, lstm_units):
    """
    构建LSTM模型
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# 可视化训练过程
def plot_training_history(history, save_path):
    """
    绘制训练过程中的损失和准确率曲线
    """
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失', color='blue')
    plt.plot(history.history['val_loss'], label='验证损失', color='orange')
    plt.title('训练与验证损失', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率', color='blue')
    plt.plot(history.history['val_accuracy'], label='验证准确率', color='orange')
    plt.title('训练与验证准确率', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# 绘制数据标签分布图
def plot_data_distribution(labels, save_path):
    """
    绘制数据标签分布图
    """
    label_counts = pd.Series(labels).value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(label_counts.index, label_counts.values, color=['green', 'red'])
    plt.title('数据标签分布', fontsize=14, fontweight='bold')
    plt.xlabel('标签', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.xticks([0, 1], ['负面情感', '正面情感'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# 绘制文本长度分布图
def plot_text_length_distribution(texts, save_path):
    """
    绘制文本长度分布图
    """
    text_lengths = [len(text.split()) for text in texts]
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('文本长度分布', fontsize=14, fontweight='bold')
    plt.xlabel('文本长度（词数）', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# 绘制词汇频率分布图
def plot_word_frequency_distribution(texts, save_path, top_n=20):
    """
    绘制词汇频率分布图
    """
    all_words = ' '.join(texts).split()
    word_freq = Counter(all_words).most_common(top_n)
    words, freqs = zip(*word_freq)

    plt.figure(figsize=(12, 7))
    plt.barh(words, freqs, color='purple')
    plt.title(f'词汇频率分布（前{top_n}个词）', fontsize=14, fontweight='bold')
    plt.xlabel('频次', fontsize=12)
    plt.ylabel('词汇', fontsize=12)
    plt.gca().invert_yaxis()  # 使频次最高的词在顶部
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# 绘制混淆矩阵图
def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    绘制混淆矩阵图
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面情感', '正面情感'],
                yticklabels=['负面情感', '正面情感'])
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# 保存模型和相关文件
def save_model_and_artifacts(model, tokenizer, history, word_index, 
                            max_len, vocab_size, embedding_dim, lstm_units,
                            X_test, y_test, y_pred, save_dir='saved_model'):
    """
    保存模型和相关文件
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建保存目录: {save_dir}")
    
    print(f"\n{'='*60}")
    print("保存模型和相关文件...")
    print('='*60)
    
    # 1. 保存完整模型
    model_path = os.path.join(save_dir, 'lstm_sentiment_model.h5')
    model.save(model_path)
    print(f"✓ 完整模型已保存: {model_path}")
    
    # 2. 保存tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer.pickle')
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Tokenizer已保存: {tokenizer_path}")
    
    # 3. 保存训练参数
    training_params = {
        'max_len': max_len,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'word_index': word_index,
        'final_val_accuracy': history.history['val_accuracy'][-1] if history else None,
        'final_val_loss': history.history['val_loss'][-1] if history else None,
        'test_accuracy': accuracy_score(y_test, y_pred),
    }
    
    params_path = os.path.join(save_dir, 'training_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(training_params, f)
    print(f"✓ 训练参数已保存: {params_path}")
    
    # 4. 保存训练历史
    if history:
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"✓ 训练历史已保存: {history_path}")
    
    # 5. 保存模型摘要
    summary_path = os.path.join(save_dir, 'model_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LSTM情感分类模型\n")
        f.write("="*50 + "\n\n")
        f.write(f"模型参数:\n")
        f.write(f"  - 最大序列长度: {max_len}\n")
        f.write(f"  - 词汇表大小: {vocab_size}\n")
        f.write(f"  - 词嵌入维度: {embedding_dim}\n")
        f.write(f"  - LSTM单元数: {lstm_units}\n\n")
        
        if history:
            f.write(f"训练结果:\n")
            f.write(f"  - 最终验证准确率: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"  - 最终验证损失: {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"  - 测试集准确率: {accuracy_score(y_test, y_pred):.4f}\n\n")
        
        f.write("分类报告:\n")
        report = classification_report(y_test, y_pred, output_dict=True)
        for key, value in report.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                f.write(f"  精确率: {value.get('precision', 0):.4f}\n")
                f.write(f"  召回率: {value.get('recall', 0):.4f}\n")
                f.write(f"  F1分数: {value.get('f1-score', 0):.4f}\n")
    
    print(f"✓ 模型摘要已保存: {summary_path}")
    
    # 6. 保存评估结果
    eval_path = os.path.join(save_dir, 'evaluation_results.txt')
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write("模型评估结果\n")
        f.write("="*50 + "\n\n")
        f.write(f"测试集大小: {len(y_test)}\n")
        f.write(f"准确率: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write("详细分类报告:\n")
        f.write(classification_report(y_test, y_pred))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        f.write(f"\n混淆矩阵:\n")
        f.write(f"           预测负面  预测正面\n")
        f.write(f"真实负面    {cm[0][0]:6d}    {cm[0][1]:6d}\n")
        f.write(f"真实正面    {cm[1][0]:6d}    {cm[1][1]:6d}\n")
    
    print(f"✓ 评估结果已保存: {eval_path}")
    print('='*60)
    print(f"所有文件已保存到目录: {save_dir}")
    print('='*60)


# 加载并使用保存的模型进行预测
def load_and_predict_example(model_dir='saved_model'):
    """
    加载保存的模型并进行示例预测
    """
    print(f"\n{'='*60}")
    print("示例：加载模型进行预测")
    print('='*60)
    
    try:
        # 加载模型
        model_path = os.path.join(model_dir, 'lstm_sentiment_model.h5')
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        
        model = load_model(model_path)
        print(f"✓ 模型加载成功: {model_path}")
        
        # 加载tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print(f"✓ Tokenizer加载成功")
        
        # 加载参数
        params_path = os.path.join(model_dir, 'training_params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        max_len = params['max_len']
        print(f"✓ 参数加载成功 (max_len={max_len})")
        
        # 示例评论
        example_reviews = [
            "这个手机质量非常好，拍照效果很清晰，运行速度很快，非常满意！",
            "物流太慢了，等了一个多星期才到货，包装也很简陋，很不满意",
            "性价比很高，功能齐全，但是电池续航一般，需要经常充电",
            "客服态度很差，问题一直不解决，售后服务让人失望",
            "设计很漂亮，手感很好，但价格有点贵，适合预算充足的人"
        ]
        
        print("\n对新的评论进行情感预测:")
        print("-" * 60)
        
        for i, review in enumerate(example_reviews, 1):
            # 预处理（需要与训练时相同的预处理）
            import jieba
            
            # 加载停用词表（这里简化处理，实际需要与训练时相同的停用词表）
            stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上'])
            
            words = jieba.lcut(review)
            filtered_words = [word for word in words if word not in stopwords]
            processed_text = ' '.join(filtered_words)
            
            # 转换为序列
            sequence = tokenizer.texts_to_sequences([processed_text])
            padded = pad_sequences(sequence, maxlen=max_len)
            
            # 预测
            prediction = model.predict(padded, verbose=0)[0][0]
            
            # 判断情感
            sentiment = "正面" if prediction > 0.5 else "负面"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            print(f"评论 {i}:")
            print(f"  内容: {review[:40]}...")
            print(f"  预测: {sentiment}情感")
            print(f"  置信度: {confidence:.2%}")
            print(f"  原始分数: {prediction:.4f}")
            print("-" * 40)
            
    except Exception as e:
        print(f"预测时发生错误: {e}")
        import traceback
        traceback.print_exc()


# 主函数
def main():
    # 配置路径
    data_path = 'E:/SUES_course/ML_Final/dataset/LSTM_data.csv'
    stopwords_path = 'E:/SUES_course/ML_Final/dataset/stop_word.txt'
    
    # 输出文件路径
    training_plot_save_path = '训练历史.png'
    data_distribution_save_path = '数据分布图.png'
    text_length_distribution_save_path = '文本长度分布图.png'
    word_frequency_distribution_save_path = '词汇频率分布图.png'
    confusion_matrix_save_path = '混淆矩阵图.png'

    # 数据加载与预处理
    print("开始数据加载与预处理...")
    texts, labels = load_and_preprocess_data(data_path, stopwords_path)

    # 绘制数据标签分布图
    print("\n绘制数据标签分布图...")
    plot_data_distribution(labels, data_distribution_save_path)
    print("数据标签分布图已保存到 {}".format(data_distribution_save_path))

    # 绘制文本长度分布图
    print("\n绘制文本长度分布图...")
    plot_text_length_distribution(texts, text_length_distribution_save_path)
    print("文本长度分布图已保存到 {}".format(text_length_distribution_save_path))

    # 绘制词汇频率分布图
    print("\n绘制词汇频率分布图...")
    plot_word_frequency_distribution(texts, word_frequency_distribution_save_path, top_n=20)
    print("词汇频率分布图已保存到 {}".format(word_frequency_distribution_save_path))

    # 文本向量化
    print("\n开始文本向量化...")
    max_len = 100
    data, tokenizer, word_index = text_vectorization(texts, max_len)

    # 划分数据集
    print("\n划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print("训练集大小：{}，测试集大小：{}".format(len(X_train), len(X_test)))

    # 模型构建
    print("\n构建模型...")
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    lstm_units = 128
    model = build_lstm_model(vocab_size, embedding_dim, lstm_units)

    # 模型训练
    print("\n开始模型训练...")
    epochs = 20
    batch_size = 64
    history = model.fit(X_train, np.array(y_train),
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    # 模型评估
    print("\n模型评估...")
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    print("分类报告：")
    print(classification_report(y_test, y_pred))
    print("准确率：", accuracy_score(y_test, y_pred))

    # 保存模型和相关文件
    save_model_and_artifacts(
        model=model,
        tokenizer=tokenizer,
        history=history,
        word_index=word_index,
        max_len=max_len,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        save_dir='saved_model'  # 保存到saved_model目录
    )

    # 绘制训练过程曲线
    print("\n绘制训练过程曲线...")
    plot_training_history(history, training_plot_save_path)
    print("训练过程曲线已保存到 {}".format(training_plot_save_path))

    # 绘制混淆矩阵图
    print("\n绘制混淆矩阵图...")
    plot_confusion_matrix(y_test, y_pred, confusion_matrix_save_path)
    print("混淆矩阵图已保存到 {}".format(confusion_matrix_save_path))

    # 运行示例预测
    load_and_predict_example('saved_model')


if __name__ == "__main__":
    main()