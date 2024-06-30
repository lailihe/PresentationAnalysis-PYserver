import sys
import re # 정규 표현식 위한 모듈
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pymongo import MongoClient
from bson import ObjectId
import jwt
from google.cloud import speech
from google.cloud import storage
import gridfs
import subprocess
import librosa
import joblib
from Hate_catch.hate_catch import detect_hate_speech_in_sentences, tokenize
from Audio_Analysis.pitch import analyze_pitch # 피치
from Audio_Analysis.volume import analyze_volume # 볼륨

# 현재 디렉터리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 벡터라이저와 모델 로드
model2 = joblib.load('Hate_catch/UPhate_speech_ensemble_model.pkl')
vectorizer2 = joblib.load('Hate_catch/UPtfidf_vectorizer.pkl')
vectorizer2.tokenizer = tokenize

# Flask 애플리케이션 생성 및 CORS 설정
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB 연결 설정
mongo_url = "mongodb+srv://ckdgml1302:admin@cluster0.cw4wxud.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_url)
db = client.get_database('test')
recordings_collection = db['recordings']
fs = gridfs.GridFS(db, collection='recordings')

# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\tpf-project-421803-edc9c4c67795.json"
GCS_BUCKET_NAME = 'tpf-bucket-01'

# JWT 비밀 키
SECRET_KEY = 'your_secret_key'

# JWT 토큰 검증 함수
def verify_token(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded['userId']
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        print("Token verification failed:", str(e))
        return None

# 오디오 파일 (확장자)변환 함수
def convert_audio(input_data, input_format='m4a', output_format='wav', sample_rate=16000):
    input_file = "input." + input_format
    output_file = "output." + output_format
    with open(input_file, 'wb') as f:
        f.write(input_data)
    subprocess.run(['ffmpeg', '-i', input_file, '-ar', str(sample_rate), '-ac', '1', output_file], check=True)
    with open(output_file, 'rb') as f:
        output_data = f.read()
    os.remove(input_file)
    os.remove(output_file)
    return output_data

# google cloud storage file upload 함수
def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    return f"gs://{bucket_name}/{destination_blob_name}"

# 텍스트에서 단어 수 계산 함수
def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)

# 말 속도 분석 함수
def analyze_speech_rate(audio_file_path, transcription_response):
    y, sr = librosa.load(audio_file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Get word count from the transcription response
    word_count = sum([count_words(result.alternatives[0].transcript) for result in transcription_response.results])
    
    word_rate = word_count / duration * 60  # words per minute
    if word_rate <= 90:
        score = "청중이 지루해 합니다!"
    elif 90 < word_rate <= 120:
        score = "청중이 이해하기 좋습니다!"
    else:
        score = "말이 빨라서 청중이 이해하기 어렵습니다!"
    
    return word_rate, score

# 침묵 시간 계산 함수
def calculate_silence_durations(transcription_response):
    silence_durations = []
    prev_end_time = None
    prev_word = None

    for result in transcription_response.results:
        for word_info in result.alternatives[0].words:
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()
            if prev_end_time is None:
                prev_word = word_info.word
                prev_end_time = end_time
                continue
            silence_duration = start_time - prev_end_time
            if silence_duration >= 1.5:
                silence_durations.append((prev_word, word_info.word, silence_duration))
            prev_word = word_info.word
            prev_end_time = end_time
    return silence_durations

# 텍스트에서 키워드 추출 함수
def extract_keywords(transcript):
    pattern = r'\b(응|음|아|어)\b'
    regex_words = re.findall(pattern, transcript)
    regex_word_counts = Counter(regex_words)
    words = re.findall(r'\b\w+\b', transcript)
    normal_words = [word for word in words if word not in regex_words]
    normal_word_counts = Counter({word: count for word, count in Counter(normal_words).items() if count >= 3})
    okt = Okt()
    nouns = okt.nouns(transcript)
    nouns_joined = ' '.join(nouns)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([nouns_joined])
    tfidf_scores = tfidf_matrix.toarray().flatten()
    tfidf_dict = {word: score for word, score in zip(vectorizer.get_feature_names_out(), tfidf_scores)}
    sorted_tfidf = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
    top_keywords = [word for word, score in sorted_tfidf[:5]]
    nouns_only = [word for word in okt.nouns(transcript)]
    keywords_nouns = [word for word in top_keywords if word in nouns_only]
    return regex_word_counts, normal_word_counts, keywords_nouns

# < 특정 녹음 파일 텍스트 반환 엔드포인트 >
@app.route('/recordings/<file_id>/transcript', methods=['GET'])
def get_transcript(file_id):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Authorization header missing'}), 403
    token = auth_header.split(' ')[1]
    user_id = verify_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid token'}), 401

    try:
        recording_file = fs.find_one({'_id': ObjectId(file_id)})
        if not recording_file:
            return jsonify({'error': 'Recording file not found'}), 404
        recording_metadata = recordings_collection.find_one({'fileId': ObjectId(file_id), 'userId': ObjectId(user_id)})
        if not recording_metadata:
            return jsonify({'error': 'Recording metadata not found'}), 404
        audio_data = recording_file.read()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    try:
        converted_audio = convert_audio(audio_data, input_format='m4a', output_format='wav', sample_rate=16000)
        with open("temp.wav", "wb") as f:
            f.write(converted_audio)
        gcs_uri = upload_to_gcs("temp.wav", GCS_BUCKET_NAME, f"{file_id}.wav")
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            enable_word_time_offsets=True,  # 단어 타임스탬프 활성화
            enable_automatic_punctuation=True  # 자동 구두점 활성화
        )
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=180)  # 타임아웃을 180초로 설정
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        word_rate, speed_score = analyze_speech_rate("temp.wav", response)
        hate_speech_results, hate_speech_ratio = detect_hate_speech_in_sentences(transcript, model2, vectorizer2)
        silence_durations = calculate_silence_durations(response)
        regex_word_counts, normal_word_counts, keywords_nouns = extract_keywords(transcript)

        # 결과 업데이트
        recordings_collection.update_one(
            {'_id': ObjectId(recording_metadata['_id'])},
            {'$set': {
                'transcript': transcript,
                'hate_speech_results': hate_speech_results,
                'hate_speech_ratio': hate_speech_ratio,
                'silence_durations': silence_durations,
                'keywords_nouns': keywords_nouns,
                'regex_word_counts': regex_word_counts,
                'normal_word_counts': normal_word_counts
            }}
        )
        os.remove("temp.wav")

        return jsonify({
            'transcript': transcript,
            'word_rate': word_rate,
            'speed_score': speed_score,
            'silence_durations': silence_durations,
            'keywords_nouns': keywords_nouns,
            'regex_word_counts': regex_word_counts,
            'normal_word_counts': normal_word_counts,
            'hate_speech_results': hate_speech_results,
            'hate_speech_ratio': hate_speech_ratio
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# < 특정 녹음 파일 피치, 볼륨 분석 결과 반환 엔드포인트>
@app.route('/recordings/<file_id>/analysis', methods=['GET'])
def get_analysis(file_id):
    auth_header = request.headers.get('Authorization') # 요청 헤더에서 Authorization 헤더를 가져옴
    if not auth_header: # Authorization 헤더 없는 경우
        return jsonify({'error': 'Authorization header missing'}), 403
    token = auth_header.split(' ')[1] # Authorization 헤더에서 토큰 추출
    user_id = verify_token(token) # 토큰 검증하고 사용자 ID 가져옴
    if not user_id: # 유효하지 않은 토큰인 경우
        return jsonify({'error': 'Invalid token'}), 401

    try:
        recording_file = fs.find_one({'_id': ObjectId(file_id)}) # 파일 ID로 GridFS에서 녹음 파일 찾기
        if not recording_file:
            return jsonify({'error': 'Recording file not found'}), 404
        recording_metadata = recordings_collection.find_one({'fileId': ObjectId(file_id), 'userId': ObjectId(user_id)}) # 파일 ID와 사용자 ID로 메타데이터 찾기
        if not recording_metadata:
            return jsonify({'error': 'Recording metadata not found'}), 404
        audio_data = recording_file.read() # 녹음 파일 데이터 읽기
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    try:
        converted_audio = convert_audio(audio_data, input_format='m4a', output_format='wav', sample_rate=16000) # 오디오 데이터를 wav 형식으로 변환
        with open("temp.wav", "wb") as f: # 임시 파일로 저장
            f.write(converted_audio)

        # Pitch 분석
        pitch_analysis = analyze_pitch("temp.wav") # 임시 파일에서 피치 분석 수행
        print("Pitch 분석 완료")

        # Volume analysis
        volume_analysis = analyze_volume("temp.wav") # 임시 파일에서 볼륨 분석 수행
        print("Volume 분석 완료")

        os.remove("temp.wav") # 임시 파일 삭제

        return jsonify({ # 분석 결과 JSON 형식으로 반환
            'pitch_analysis': pitch_analysis,
            'volume_analysis': volume_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
