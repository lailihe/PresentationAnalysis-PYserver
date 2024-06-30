import re
from konlpy.tag import Okt

# 형태소 분석기 로드
okt = Okt()

# 텍스트 전처리: 공백 표준화 및 특수문자 제거
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 여러 개의 공백을 하나의 공백으로 표준화
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]', '', text)  # 한글, 숫자, 영문, 공백 외의 문자 제거
    return text.strip()  # 문자열 양 끝의 공백 제거

# 형태소 분석기를 사용하여 텍스트 토큰화
def tokenize(text):
    return okt.morphs(text)  # 형태소 분석기 이용해 토큰화

# 혐오 표현 탐지 함수: 개별 토큰에 대한 예측 확률을 기반으로 혐오 표현 토큰과 확률을 반환
def detect_hate_speech_model2(text, model, vectorizer, threshold=0.7):
    cleaned_text = clean_text(text)  # 텍스트 전처리
    tokens = tokenize(cleaned_text)  # 텍스트 토큰화
    hate_speech_tokens = []  # 혐오 표현 토큰
    probabilities = []  # 각 토큰의 예측 확률

    for token in tokens:
        token_str = ' '.join([token])  # 토큰을 문자열로 변환
        vectorized_text = vectorizer.transform([token_str])  # 벡터화
        prediction_proba = model.predict_proba(vectorized_text)[0][1]  # 혐오 표현 예측 확률 계산

        if prediction_proba >= threshold:  # 예측 확률이 임계값 이상인 경우
            hate_speech_tokens.append(token)  # 혐오 표현 토큰 추가
            probabilities.append(prediction_proba)  # 해당 확률 추가

    return hate_speech_tokens, probabilities  # 혐오 표현 토큰과 확률 반환

# 문장 단위로 혐오 표현 탐지 및 통계 계산
def detect_hate_speech_in_sentences(text, model, vectorizer):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 문장을 마침표, 느낌표, 물음표로 분리
    results = []  # 결과 리스트
    hate_speech_count = 0  # 혐오 표현 단어 수 카운트
    total_word_count = 0  # 전체 단어 수 카운트

    for sentence in sentences:
        hate_speech_tokens2, probabilities2 = detect_hate_speech_model2(sentence, model, vectorizer)  # 각 문장에서 혐오 표현 탐지

        if hate_speech_tokens2:  # 혐오 표현이 있는 경우
            results.append({
                "sentence": sentence,
                "model2_results": list(zip(hate_speech_tokens2, probabilities2))  # 문장, 혐오 표현 토큰 및 확률 저장
            })
            hate_speech_count += len(hate_speech_tokens2)  # 혐오 표현 단어 수 추가
        total_word_count += len(tokenize(sentence))  # 전체 단어 수 추가

    hate_speech_ratio = (hate_speech_count / total_word_count) * 100 if total_word_count > 0 else 0  # 혐오 표현 비율 계산
    
    return results, hate_speech_ratio  # 결과 및 비율 반환
