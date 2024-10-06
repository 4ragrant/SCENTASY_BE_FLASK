from langchain_community.vectorstores import FAISS

def create_vectorstore(documents, embeddings):
    # 문서 분할 및 벡터스토어 생성
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        split_documents.extend([{"text": chunk} for chunk in chunks])

    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_texts([doc["text"] for doc in split_documents], embedding=embeddings)
    print("벡터스토어가 생성되었습니다.")
    return vectorstore
