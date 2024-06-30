# 혐오 표현 감지 기능을 모듈화 하여 호출 사용 가능
import joblib
import re
from konlpy.tag import Okt

# 형태소 분석기 로드
okt = Okt()

def clean_text(text):
    # 텍스트 전처리: 공백을 표준화 - 여러개의 공백을 하나의 공백으로
    # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]', '', text)
    return text.strip()

# 형태소 분석기를 사용하여 텍스트 토큰화
def tokenize(text):
    return okt.morphs(text)

# 저장된 모델과 벡터라이저 로드
model2 = joblib.load('UPhate_speech_ensemble_model.pkl')
vectorizer2 = joblib.load('UPtfidf_vectorizer.pkl')

# 벡터라이저의 tokenizer 재설정
vectorizer2.tokenizer = tokenize

def detect_hate_speech_model2(text, model, vectorizer, threshold=0.7):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    hate_speech_tokens = []
    probabilities = []

    for token in tokens:
        token_str = ' '.join([token])
        vectorized_text = vectorizer.transform([token_str])
        prediction_proba = model.predict_proba(vectorized_text)[0][1]
        
        if prediction_proba >= threshold:
            hate_speech_tokens.append(token)
            probabilities.append(prediction_proba)
    
    return hate_speech_tokens, probabilities

def detect_hate_speech_in_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []

    for sentence in sentences:
        hate_speech_tokens2, probabilities2 = detect_hate_speech_model2(sentence, model2, vectorizer2)
        
        if hate_speech_tokens2:
            results.append({
                "sentence": sentence,
                "model2_results": list(zip(hate_speech_tokens2, probabilities2))
            })
    
    return results

# 함수 호출 예제
if __name__ == "__main__":
    P_text = "오늘은 여행에 대해서 발표하겠습니다. 현재 한국에는 많은 짱개가 있습니다. 이런 짱개같은 좌좀들은 전부 치워버려야 합니다. 제 친구는 어제 귀국했습니다. 그는 하루빨리 이 문제를 해결해야한다고 합니다. 공항에서 지나가는 사람들 중 일부 여자들은 동남아 냄새 나게 생겼다며 불평을 하기도 했습니다. 최악입니다."
    
    results = detect_hate_speech_in_sentences(P_text)

    for result in results:
        print("\n<<모델 (앙상블 모델)>>")
        
        print(f"문장: {result['sentence']}")
        if result["model2_results"]:
            for token, prob in result["model2_results"]:
                print(f"단어: {token}, 혐오 정도: {prob:.2f}")
        else:
            print("혐오 표현이 감지되지 않았습니다.")
