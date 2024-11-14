import re
import pandas as pd
from kiwipiepy import Kiwi

# 불용어 리스트
stopwords = ["의", "이", "가", "는", "에", "들", "을", "로", "과", "를", "와", "한", "하다"]

# 동의어 데이터 로드
synonym_file_path = 'data/synonym_data.csv'
synonym_data = pd.read_csv(synonym_file_path, encoding='utf-8')
synonym_dict = {row['Category']: [synonym.strip() for synonym in row['Synonyms'].split(',')] for index, row in
                synonym_data.iterrows()}

# Kiwi 형태소 분석기 사용
class KiwiProcessor:
    def __init__(self):
        self.kiwi = Kiwi()

    # 유사도 계산 함수
    def calculate_similarity_with_synonyms(self, summary_sentence):
        sentence = re.sub(r'[^\w\s]', '', summary_sentence)  # 특수 문자 제거
        tokens = [word[0] for word in self.kiwi.tokenize(sentence)]  # 형태소 분석
        words = [word for word in tokens if word not in stopwords]  # 불용어 제거
        sentence = ' '.join(words)  # 정제된 단어를 다시 문장으로 결합

        similarity_results = {}
        for category, synonyms in synonym_dict.items():
            match_count = sum([1 for synonym in synonyms if synonym in sentence])
            similarity_score = match_count / len(synonyms) if len(synonyms) > 0 else 0
            similarity_results[category] = similarity_score

        return similarity_results

# 가중치 적용 함수
import math

# 가중치 적용 함수
def apply_weights_to_similarity(similarity_results, base_weight=0, category_weights=None, weight_factor=5,
                                detailed_adjustment=False):
    weighted_results = {}

    if category_weights is None:
        category_weights = {category: 1 for category in similarity_results.keys()}

    for category, similarity in similarity_results.items():
        category_weight = category_weights.get(category, 1)
        if detailed_adjustment:
            if similarity > 0.8:
                weight = category_weight + similarity * weight_factor * 1000
            elif similarity > 0.4:
                weight = category_weight + similarity * weight_factor * 500
            elif similarity > 0.1:
                weight = category_weight + similarity * weight_factor * 300
            else:
                weight = base_weight
        else:
            weight = base_weight + (similarity * weight_factor)

        # 소수점 첫 번째 자리까지만 출력하고, 두 번째 자리에서 반올림
        rounded_weight = math.ceil(weight * 10) / 10.0
        weighted_results[category] = rounded_weight

    return weighted_results

