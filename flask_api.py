from flask import Flask, request, jsonify
from prompt_template import create_prompt
from prompt_template import create_title_description_prompt
from chat_model import create_chain
from kiwi_module import KiwiProcessor, stopwords, apply_weights_to_similarity
import torch.nn as nn
import threading
import torch
import pandas as pd
import joblib
import numpy as np

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


app = Flask(__name__)

# 락 생성
store_lock = threading.Lock()

# 세션 기록을 저장할 딕셔너리
store = {}

# 프롬프트 생성
prompt = create_prompt()

# `TransformerModel` 클래스 정의
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, hidden_dim, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.pos_encoder(x)
        x = self.transformer(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        return torch.sigmoid(self.output_layer(x))

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
accord_scaler = None
accord_model = None

# 하이퍼파라미터
input_size = 52  # 입력 피처 수
output_size = 26  # 예측할 노트 수
num_heads = 4
hidden_dim = 128
num_layers = 3

def load_models():
    global model, scaler, accord_model, accord_scaler

    # 스케일러 로드
    scaler = joblib.load('scalers/new_scaler.pkl')
    #scaler = StandardScaler()
    accord_scaler = joblib.load('scalers/accord_scaler.pkl')

    # 전체 모델을 로드
    model = TransformerModel(input_size, output_size, num_heads, hidden_dim, num_layers)
    model.load_state_dict(torch.load('models/new_trained_model.pth'))
    accord_model = torch.load('models/accord_model.pth')

    model.eval()
    accord_model.eval()

load_models()

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
    session_id = data.get('sessionId', 'default_session')

    # 대화 체인 생성
    rag_with_history = create_chain_with_history()

    # 대화 처리
    response = rag_with_history.invoke(
        {"question": question},
        {"configurable": {"session_id": session_id}}
    )

    # 대화 기록 저장
    store_conversation(session_id, question, response)

    print(f"Updated store for session ID: {session_id} - {store[session_id]}")

    return jsonify({'response': response})

# 유사도 계산 및 예측 API
@app.route("/api/recipe", methods=['POST'])
def similarity_and_predict():
    data = request.json
    print(f"Received data: {data}")  # 요청 데이터 로그
    session_id = data.get('sessionId')
    #session_id = data.get('sessionId', 'default_session')

    print(f"Requested session ID: {session_id}")

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
        
        # 예측 호출
        predicted_notes = predict_internal(weighted_results) # 이진값
        predicted_notes_array = [int(x.strip()) for x in predicted_notes.split(',')]
        predicted_accords = predict_accords(predicted_notes_array)

        # 이름 매핑
        predicted_note_names = map_notes_to_columns(predicted_notes_array)
        predicted_accords_with_columns = map_accords_to_columns(predicted_accords)

        # 제목과 설명 생성
        title, description = generate_title_and_description(conversation_text, predicted_note_names, predicted_accords_with_columns)

        return jsonify({
            'input_data': weighted_results,
            'predicted_notes': predicted_notes,
            'predicted_accords': predicted_accords_with_columns,
            'title': title,
            'description': description
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 내부 예측 함수
def predict_internal(weighted_results):
    # 가중치를 DataFrame으로 변환
    input_df = pd.DataFrame([weighted_results])

    # 스케일러 적용
    input_scaled = scaler.transform(input_df.values)

    # 텐서로 전환
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # 예측
    with torch.no_grad():
        predicted_notes = model(input_tensor)

    # 예측 데이터 이진화
    predicted_notes_binary = (predicted_notes > 0.4).int().numpy().tolist()
    # 이진화 (임계값 0.5 적용)
    #predicted_notes_binary = np.zeros_like(predicted_notes)  # 모든 값을 0으로 초기화
    #top_5_indices = np.argsort(predicted_notes[0])[::-1][:5]  # 상위 5개 인덱스 찾기
    #predicted_notes_binary[0, top_5_indices] = 1  # 상위 5개만 1로 설정

    # # 후처리 과정: Accords_floral이 0인 경우, 플로럴 관련 노트를 0으로 설정
    # if weighted_results.get('Accords_floral', 0) == 0:
    #     floral_note_indices = [8, 9, 12, 13]  # Freesia, Rose, Muguet, Magnolia
    #     for idx in floral_note_indices:
    #         predicted_notes_binary[0][idx] = 0
    #
    # # 후처리 과정: Accords_fruity가 0인 경우, Fruity 관련 노트를 0으로 설정
    # if weighted_results.get('Accords_fruity', 0) == 0:
    #     fruity_note_indices = [16, 4, 5, 6]  # Black Currant, Peach, Fig, Black Cherry
    #     for idx in fruity_note_indices:
    #         predicted_notes_binary[0][idx] = 0

    # 결과 배열을 문자열로 변환
    predicted_notes_string = ', '.join(map(str, predicted_notes_binary[0]))

    return predicted_notes_string

# 노트 매핑 함수
def map_notes_to_columns(predicted_notes):
    note_columns = [
        "Bergamot_TopNote", "Mint_TopNote", "Lemon_TopNote", "Aqual_TopNote",
        "Grapefruit Blossom_TopNote", "Peach_TopNote", "Fig_TopNote",
        "Black Cherry_TopNote", "Green_TopNote", "Freesia_MiddleNote",
        "Rose_MiddleNote", "Pepper_MiddleNote", "Rosemary_MiddleNote",
        "Muguet_MiddleNote", "Magnolia_MiddleNote", "Ocean_MiddleNote",
        "Black Currant_MiddleNote", "Musk_MiddleNote", "Vanilla_BaseNote",
        "Sandalwood_BaseNote", "Leather_BaseNote", "Patchouli_BaseNote",
        "Cedar_BaseNote", "Abmer_BaseNote", "Frankincense_BaseNote",
        "Hinoki Wood_BaseNote"
    ]

    active_notes = [note_columns[i] for i, value in enumerate(predicted_notes) if value == 1]
    return active_notes

# 어코드 예측 함수
def predict_accords(predicted_notes):
    # 입력 데이터 검증
    if not predicted_notes:
        return jsonify({'error': 'Predicted notes are required.'}), 400

    # 예측된 노트를 데이터프레임으로 변환
    input_df = pd.DataFrame([predicted_notes])

    # 스케일러 적용
    input_scaled = accord_scaler.transform(input_df)

    # 텐서로 전환
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # 어코드 예측
    with torch.no_grad():
        predicted_accords = accord_model(input_tensor)

    # 어코드 비율을 리스트로 반환
    predicted_accords_ratios = predicted_accords.numpy().tolist()[0]

    return predicted_accords_ratios

# 어코드 매핑 함수
def map_accords_to_columns(predicted_accords):
    accord_columns = [
        "시트러스", "구르망", "오리엔탈", 
        "머스크", "프루티", "우디", 
        "스파이시", "플로럴", "아쿠아틱", 
        "레더", "아로마틱", "스모키"
    ]

    # 어코드 이름과 값을 매핑 및 필터링
    predicted_accords_with_columns = [
        {"accord": accord_columns[i], "value": round(predicted_accords[i], 3)} 
        for i in range(len(predicted_accords)) if predicted_accords[i] >= 0.5
    ]

    return predicted_accords_with_columns

# 제목과 설명 생성 함수
def generate_title_and_description(conversation_text, predicted_note_names, predicted_accords_with_columns):
    # ChatOpenAI 인스턴스 생성
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    prompt_template = create_title_description_prompt()

    notes_str = ', '.join(predicted_note_names)
    accords_str = ', '.join([f"{accord['accord']} ({accord['value']})" for accord in predicted_accords_with_columns])

    print(notes_str)
    print(accords_str)

    prompt = prompt_template.format(
        conversation_text=conversation_text,
        notes_str=notes_str,
        accords_str=accords_str
    )
    
    messages = [
        SystemMessage(content="You are an assistant that generates creative perfume titles and descriptions."),
        HumanMessage(content=prompt)
    ]

    # 모델 호출
    response = llm(messages)

    # 응답에서 제목과 설명 추출
    generated_text = response.content.strip().split("\n")

    title = generated_text[0].replace("제목: ", "").strip() if len(generated_text) > 0 and generated_text[0].strip() else "우디한 성공의 순간"
    description = generated_text[1].replace("설명: ", "").strip() if len(generated_text) > 1 and generated_text[1].strip() else (
        "중요한 발표를 앞둔 당신을 위한 향입니다. 상쾌한 레몬의 탑 노트로 시작하여, 중간에는 프리지아와 페퍼의 조합이 "
        "당신에게 활력을 불어넣어줍니다. 마지막으로, 샌달우드와 패츌리의 베이스 노트가 깔끔하고 우아한 향으로 마무리되며, "
        "가을의 따뜻한 우디한 느낌을 전해줍니다. 이 향은 당신의 중요한 순간을 더욱 특별하게 만들어줄 것입니다."
    )
    
    return title, description

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)