from flask import Flask, request, jsonify
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from prompt_template import create_prompt
from chat_model import create_chain
from kiwi_module import KiwiProcessor, stopwords, apply_weights_to_similarity
import torch.nn as nn
import threading
import torch
import pandas as pd
import joblib

app = Flask(__name__)

# 락 생성
store_lock = threading.Lock()

# 세션 기록을 저장할 딕셔너리
store = {}

# 프롬프트 생성
prompt = create_prompt()

# `MLP` 클래스 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        current_input_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        self.output_layer = nn.Linear(current_input_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

# Load pre-trained model and scaler
scaler = None
model = None

def load_model():
    global model, scaler

    # 스케일러 로드
    scaler = joblib.load('scaler.pkl')

    # 전체 모델을 로드
    model = torch.load('model.pth')
    model.eval()

load_model()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id):
    with store_lock:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()  # ChatMessageHistory 객체로 변경
        return store[session_id]

# 대화 체인 생성
def create_chain_with_history():
    chain, memory = create_chain(prompt)

    # RunnableWithMessageHistory 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 질문 키
        history_messages_key="chat_history",  # 대화 기록 키
    )
    return rag_with_history

# 대화 기록 저장 함수
def store_conversation(session_id, user_input, assistant_response):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # 새로운 세션 기록 생성
    # 메시지 기록 (사용자의 메시지와 어시스턴트의 응답)
    store[session_id].add_message(HumanMessage(content=user_input))  # 사용자 메시지 추가
    store[session_id].add_message(AIMessage(content=assistant_response))  # 어시스턴트의 응답 추가

# 대화 처리 API
@app.route("/api/chats", methods=['POST'])
def chat():
    data = request.json
    question = data.get('input', '')
    session_id = data.get('session_id', 'default_session')

    # 대화 체인 생성
    rag_with_history = create_chain_with_history()

    # 대화 처리
    response = rag_with_history.invoke(
        {"question": question},
        {"configurable": {"session_id": session_id}}
    )

    # 대화 기록 저장
    store_conversation(session_id, question, response)

    return jsonify({'response': response})

# 유사도 계산 API
@app.route("/api/similarity", methods=['POST'])
def similarity():
    data = request.json
    session_id = data.get('session_id', 'default_session')

    # 해당 세션의 대화 기록을 가져옴
    if session_id not in store:
        return jsonify({'error': 'No conversation history for this session.'}), 404

    conversation_history = store[session_id]

    # 대화 기록에서 실제 메시지를 추출
    try:
        conversation_text = ""
        for message in conversation_history.messages:
            if isinstance(message, HumanMessage):
                conversation_text += f"User: {message.content} "
            elif isinstance(message, AIMessage):
                conversation_text += f"Assistant: {message.content} "

        # 유사도 계산
        kiwi_processor = KiwiProcessor()
        similarity_results = kiwi_processor.calculate_similarity_with_synonyms(conversation_text)

        # 가중치 적용
        weighted_results = apply_weights_to_similarity(similarity_results)

        return jsonify({'input_data': weighted_results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 예측 API
@app.route("/api/predict", methods=['POST'])
def predict():
    data = request.json
    weighted_results = data.get('input_data')

    # 가중치를 DataFrame으로 변환
    input_df = pd.DataFrame([weighted_results])

    # 스케일러 적용
    input_scaled = scaler.transform(input_df)

    # 텐서로 전환
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # 예측
    with torch.no_grad():
        predicted_notes = model(input_tensor)

    # 예측 데이터 이진화
    predicted_notes_binary = (predicted_notes > 0.4).int().numpy().tolist()

    # 후처리 과정: Accords_floral이 0인 경우, 플로럴 관련 노트를 0으로 설정
    if weighted_results.get('Accords_floral', 0) == 0:
        floral_note_indices = [8, 9, 12, 13]  # Freesia, Rose, Muguet, Magnolia
        for idx in floral_note_indices:
            predicted_notes_binary[0][idx] = 0

    # 후처리 과정: Accords_fruity가 0인 경우, Fruity 관련 노트를 0으로 설정
    if weighted_results.get('Accords_fruity', 0) == 0:
        fruity_note_indices = [16, 4, 5, 6]  # Black Currant, Peach, Fig, Black Cherry
        for idx in fruity_note_indices:
            predicted_notes_binary[0][idx] = 0

    return jsonify({"predicted_notes": predicted_notes_binary[0]})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)