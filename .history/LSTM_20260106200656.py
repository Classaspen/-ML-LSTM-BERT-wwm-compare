# 首先，在文件开头导入保存模型需要的模块（如果有的话）
from tensorflow.keras.models import load_model

# 然后在主函数中，模型训练后添加：

# 模型训练
print("\n开始模型训练...")
epochs = 20
batch_size = 64
history = model.fit(X_train, np.array(y_train),
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)

# ==================== 新增：保存模型 ====================
print("\n保存训练好的模型...")

# 创建models文件夹保存所有模型相关文件
if not os.path.exists('models'):
    os.makedirs('models')

# 1. 保存完整模型（.h5格式）
model.save('models/lstm_sentiment_model.h5')
print("✓ 完整模型已保存: models/lstm_sentiment_model.h5")

# 2. 保存tokenizer（用于后续预测时相同的文本处理）
import pickle
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Tokenizer已保存: models/tokenizer.pickle")

# 3. 保存训练参数
training_info = {
    'max_len': max_len,
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'lstm_units': lstm_units,
    'word_index': word_index,
    'accuracy': history.history['val_accuracy'][-1]  # 最后一代的验证准确率
}

with open('models/training_info.pkl', 'wb') as f:
    pickle.dump(training_info, f)
print("✓ 训练参数已保存: models/training_info.pkl")

# 4. 保存模型架构图（可视化）
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='models/model_architecture.png', show_shapes=True)
print("✓ 模型架构图已保存: models/model_architecture.png")
# ==================== 新增结束 ====================

# 模型评估
print("\n模型评估...")