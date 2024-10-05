import pandas as pd
from prompt_template import create_prompt
from chat_model import create_chain

# CSV 파일 로드
file_path = 'review_done.csv'
data = pd.read_csv(file_path)

# 필요한 데이터 추출
descriptions = data['Description'].dropna().tolist()
situations = data['situation'].dropna().tolist()
situation_keywords = data['situation_keyword'].dropna().tolist()
combined_texts = descriptions + situations + situation_keywords
documents = [{"text": text} for text in combined_texts]

#임베딩 생성
#embeddings = create_embeddings()

# 벡터스토어 생성
#vectorstore = create_vectorstore(documents, embeddings)

# 검색기(Retriever) 생성
#retriever = vectorstore.as_retriever()

# 프롬프트 생성
prompt = create_prompt()

# 체인 및 메모리 생성
chain, memory = create_chain(prompt)

# 대화 예시
memory.save_context(
    {
        "input": "question",
        "chat_history": "chat_history"
    },
    {
        "output_key": "answer"
    }
)

print('끝!')
