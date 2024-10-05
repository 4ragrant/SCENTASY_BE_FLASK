from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def create_chain(prompt):
    # LLM 및 체인 생성
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # 메모리 생성
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # 체인 생성
    chain = (
        {
            #"context": itemgetter("question") | retriever,
            "context": itemgetter("question"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, memory
