import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# 데이터셋 로드
df = pd.read_csv('cefr_leveled_texts.csv')

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터 전처리
max_length = 128  # 문장의 최대 길이
texts = df['text'].tolist()
labels = df['label'].tolist()

# BERT 입력 형식에 맞게 데이터 변환
def encode_data(texts, labels):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='tf')
    encoded['labels'] = labels
    return encoded

# 데이터셋 나누기
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 데이터 인코딩
train_encodings = encode_data(train_texts, train_labels)
test_encodings = encode_data(test_texts, test_labels)

# BERT 모델 불러오기
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# 모델 학습
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
model.fit(train_encodings, epochs=3, batch_size=32)

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_encodings)
print('Test Accuracy:', test_accuracy)
