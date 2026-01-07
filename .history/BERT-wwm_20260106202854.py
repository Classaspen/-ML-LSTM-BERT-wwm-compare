import paddlehub as hub
import pandas as pd
import numpy as np

# 1. 加载预训练模型
print("加载RoBERTa模型...")
module = hub.Module(name="roberta-wwm-ext-large", version='2.0.2')

# 2. 准备你的数据集（假设CSV格式）
data = pd.read_csv('E:/SUES_course/ML_Final/dataset/LSTM_data.csv')
texts = data['review'].tolist()[:100]  # 取前100条测试，避免太长

print(f"加载 {len(texts)} 条评论进行推理...")

# 3. 执行情感分析推理
results = module.sentiment_classify(
    texts=texts,
    use_gpu=False,  # 如果GPU可用可以设为True
    batch_size=32   # 批处理大小
)

# 4. 输出结果
print("\n情感分析结果（前10条）：")
print("=" * 80)
for i, (text, result) in enumerate(zip(texts[:10], results[:10])):
    print(f"评论 {i+1}: {text[:50]}...")
    print(f"  情感: {result['sentiment_label']} (正面/负面)")
    print(f"  置信度: 正面 {result['positive_probs']:.4f}, 负面 {result['negative_probs']:.4f}")
    print(f"  预测结果: {'正面' if result['sentiment_key'] == 'positive' else '负面'}")
    print("-" * 80)

# 5. 统计结果
positive_count = sum(1 for r in results if r['sentiment_key'] == 'positive')
negative_count = len(results) - positive_count

print(f"\n统计结果:")
print(f"总评论数: {len(results)}")
print(f"正面评论: {positive_count} ({positive_count/len(results)*100:.1f}%)")
print(f"负面评论: {negative_count} ({negative_count/len(results)*100:.1f}%)")

# 6. 保存结果到CSV（可选）
if len(texts) > 0:
    result_df = pd.DataFrame(results)
    result_df.insert(0, '原始评论', texts)
    result_df.to_csv('roberta_sentiment_results.csv', index=False, encoding='utf-8')
    print(f"\n完整结果已保存到: roberta_sentiment_results.csv")