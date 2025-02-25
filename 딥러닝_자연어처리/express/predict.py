from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

text_data="""Soft as the voice of an angel\n
Breathing a lesson unhead\n
Hope with a gentle persuasion\n
Whispers her comforting word\n
Wait till the darkness is over\n
Wait till the tempest is done\n
Hope for sunshine tomorrow\n
After the shower
"""

# 첫 번째 단계는 텍스트를 정수로 인코딩하는 것입니다.
# 소스 텍스트의 각 소문자 단어에는 고유 한 정수가 할당되며 단어 시퀀스를 정수 시퀀스로 변환 할 수 있습니다.
# Keras는 이 인코딩을 수행하는 데 사용할 수있는 Tokenizer 클래스를 제공합니다 . 첫째, Tokenizer는 단어에서 고유 한 정수로의 매핑을 개발하기 위해 소스 텍스트에 적합합니다. 그런 다음 texts_to_sequences () 함수를 호출하여 텍스트 시퀀스를 정수 시퀀스로 변환 할 수 있습니다 .
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
encoded = tokenizer.texts_to_sequences([text_data])[0]
print(encoded)

# 나중에 모델에서 단어 임베딩 레이어 를 정의하고 원 핫 인코딩을 사용하여 출력 단어를 인코딩 하기 위해 어휘의 크기를 알아야합니다 .
# 단어의 크기는 word_index 속성 에 액세스하여 훈련 된 Tokenizer에서 검색 할 수 있습니다 .
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
print(tokenizer.word_index)


# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)
print(sequences)
print('총 시퀀스 개수: %d' % len(sequences))

# split into X and y elements
sequences = np.array(sequences)
X, y = sequences[:,0],sequences[:,1]
print("X=", X)
print("y=", y)

# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
print("y=", y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)

test_text = 'Wait'
print(test_text, "=>", end=" ")
encoded = tokenizer.texts_to_sequences([test_text])[0]
encoded = np.array(encoded)
output = np.argmax(model.predict(encoded), axis=-1)
for word, index in tokenizer.word_index.items():
	if index == output:
		print(word)