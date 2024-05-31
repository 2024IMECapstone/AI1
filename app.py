from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

app = Flask(__name__)

# YAMNet 모델 로드 및 클래스 이름 리스트 설정
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_names = ["Crying, sobbing", "Baby cry, infant cry", "Babbling"]  

saved_model_path="./yamnet81model/"
reloaded_model = tf.saved_model.load(saved_model_path)

# 원하는 클래스 설정
desired_class = ["Crying, sobbing", "Baby cry, infant cry", "Babbling"]

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"} ), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        # 파일을 librosa를 사용하여 로드
        wav, sample_rate = librosa.load(file, sr=16000, mono=True)

        # YAMNet 모델을 사용하여 예측
        scores, embeddings, spectrogram = yamnet_model(wav)
        scores_np = scores.numpy()
        top_classes_indices = scores_np.mean(axis=0).argsort()[-5:][::-1]

        found_desired_class = False
        detected_class = None
        for index in top_classes_indices:
            inferred_class = class_names[index]
            if inferred_class in desired_class:
                found_desired_class = True
                detected_class = inferred_class
                break

        if found_desired_class:
            my_classes=['asphyxia','hunger','normal','pain','tired','discomfort']
            reloaded_results = reloaded_model(wav)
            crying_type = my_classes[tf.math.argmax(reloaded_results)]
            print(f'The main sound is: {crying_type}')
            return jsonify({"status": "detected", "class": detected_class, "cryingType" : crying_type}), 200  #여기서 detected_class는 안 보내줘도 되기는 함
        else:
            return jsonify({"status": "not_detected"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 파일을 저장 임시 경로 생성
        temp_file_path = os.path.join("C:/Capstone/temp_videos", file.filename)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        file.save(temp_file_path)

        # 동영상 파일을 처리 및 결과를 반환
        from inference2 import process_video_file
        results_summary = process_video_file(temp_file_path)

        # 임시 파일 삭제
        os.remove(temp_file_path)

        return jsonify({"status": "processed", "results": results_summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
