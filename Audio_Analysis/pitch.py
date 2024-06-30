import numpy as np
import librosa

# 피치 값을 추출하는 함수
def extract_pitch(audio_file):
    # 오디오 파일 로드하여 오디오 시간 데이터(y), 샘플링 속도(sr) 반환
    y, sr = librosa.load(audio_file)
    # 오디오 피치(목소리 높낮이)와 그 진폭(크기) 계산
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    pitch_values = [] # 피치 값 저장 리스트
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax() # 가진 큰 진폭 찾기
        pitch = pitches[index, t] # 해당 피치 값 가져옴
        if pitch > 0: # 0보다 큰 경우만 리스트에 추가
            pitch_values.append(float(pitch))
    
    return pitch_values

# 피치 사용 범위를 계산하는 함수
def calculate_pitch_usage(pitch_values):
    # 각 범위에 해당하는 피치 값 개수 계산하여 딕셔너리에 저장
    usage = {
        'low': int(np.sum(np.array(pitch_values) <= 85)),
        'slightly_low': int(np.sum((np.array(pitch_values) > 85) & (np.array(pitch_values) <= 125))),
        'medium': int(np.sum((np.array(pitch_values) > 125) & (np.array(pitch_values) <= 180))),
        'slightly_high': int(np.sum((np.array(pitch_values) > 180) & (np.array(pitch_values) <= 255))),
        'high': int(np.sum(np.array(pitch_values) > 255))
    }
    return usage # 계산된 피치 사용 범위 반환

# 피치 값을 기준으로 점수를 부여하고, 이를 백분율로 변환하는 함수
def calculate_pitch_score(pitch_values):
    scores = []
    for value in pitch_values:
        if value > 255:
            scores.append(5)
        elif value > 180:
            scores.append(4)
        elif value > 125:
            scores.append(3)
        elif value > 85:
            scores.append(2)
        else:
            scores.append(1)
    total_score = sum(scores) # 모든 점수 합산
    max_score = len(pitch_values) * 5 # 최대 점수: 피치 값 개수*5
    percentage_score = (total_score / max_score) * 100 # 총점수 백분율 계산
    return percentage_score, scores # 백분열점수, 각 피치 값 점수

# 메인 함수
def analyze_pitch(audio_file):
    pitch_values = extract_pitch(audio_file) # 피치 값 추출
    pitch_usage = calculate_pitch_usage(pitch_values) # 피치 사용 범위 계산
    pitch_score, scores = calculate_pitch_score(pitch_values) # 피치 점수 계산
    print(f"추출된 피치값: {pitch_values}")
    print(f"피치 사용 범위: {pitch_usage}")
    print(f"피치 점수: {pitch_score}, 첫 10개의 Scores: {scores[:10]}")
    return {
        'pitch_values': [float(value) for value in pitch_values],
        'pitch_usage': pitch_usage,
        'pitch_score': float(pitch_score),
        'scores': [int(score) for score in scores]
    }

# 테스트용 메인
if __name__ == "__main__":
    audio_file = 'path_to_audio_file.wav'
    result = analyze_pitch(audio_file)
    print(result)
