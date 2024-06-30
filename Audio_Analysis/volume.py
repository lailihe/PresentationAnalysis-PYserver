import numpy as np
import librosa

# 볼륨 값을 추출하는 함수
def calculate_rms(audio_file):
    y, sr = librosa.load(audio_file)  # sr은 사용되지 않지만 librosa.load 함수 반환 기본값
    frame_length = 2048  # 프레임 길이 설정
    hop_length = 512  # hop 길이 설정
    # RMS(루트 평균 제곱) 에너지 계산
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms

# RMS 값을 dB SPL 단위로 변환하는 함수
def rms_to_db_spl(rms_values):
    reference = 20e-6  # 20 μPa 기준 값
    db_spl_values = 20 * np.log10(rms_values / reference + 1e-10)
    db_spl_values[db_spl_values < 0] = 0  # 음수 값을 0으로 변환
    return db_spl_values

# 볼륨 사용 범위를 계산하는 함수
def calculate_volume_usage(db_values):
    usage = {
        'low': int(np.sum((db_values < 40))),
        'slightly_low': int(np.sum((db_values >= 40) & (db_values < 60))),
        'medium': int(np.sum((db_values >= 60) & (db_values < 75))),
        'slightly_high': int(np.sum((db_values >= 75) & (db_values < 85))),
        'high': int(np.sum((db_values >= 85)))
    }
    return usage

# 볼륨 점수를 계산하는 함수
def calculate_volume_score(db_values):
    scores = []  # dB 값을 점수로 변환하여 저장할 리스트
    for value in db_values:
        if value >= 85:
            scores.append(5)
        elif value >= 75:
            scores.append(4)
        elif value >= 60:
            scores.append(3)
        elif value >= 40:
            scores.append(2)
        else:
            scores.append(1)
    total_score = sum(scores)
    max_score = len(db_values) * 5
    percentage_score = (total_score / max_score) * 100
    return percentage_score, scores  # 백분율 점수, 각 dB 값의 점수

# 메인 함수
def analyze_volume(audio_file):
    try:
        rms_values = calculate_rms(audio_file)  # 오디오 파일에서 RMS 값 추출
        print(f"추출된 RMS 첫 10개 values: {rms_values[:10]}")
    except Exception as e:
        print(f"Error in calculate_rms: {str(e)}")
        return None

    try:
        db_spl_values = rms_to_db_spl(rms_values)  # RMS 값을 dB SPL로 변환
        print(f"추출된 dB SPL 첫 10개 values: {db_spl_values[:10]}")
    except Exception as e:
        print(f"Error in rms_to_db_spl: {str(e)}")
        return None

    try:
        volume_usage = calculate_volume_usage(db_spl_values)  # 볼륨 사용 범위 계산
        print(f"계산된 볼륨 사용 범위: {volume_usage}")
    except Exception as e:
        print(f"Error in calculate_volume_usage: {str(e)}")
        return None

    try:
        volume_score, scores = calculate_volume_score(db_spl_values)
        print(f"Volume score: {volume_score}, Scores: {scores[:10]}")  # 로그 추가
    except Exception as e:
        print(f"Error in calculate_volume_score: {str(e)}")
        return None

    return {
        'rms_values': [float(value) for value in rms_values],
        'db_values': [float(value) for value in db_spl_values],
        'volume_usage': volume_usage,
        'volume_score': float(volume_score),
        'scores': [int(score) for score in scores]
    }

# 테스트용 메인
if __name__ == "__main__":
    audio_file = 'path_to_audio_file.wav'
    result = analyze_volume(audio_file)
    print(result)
