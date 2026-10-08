cat << 'EOF' > analyze_motion.py
import argparse
import os
import urllib.request
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime が見つかりません。")

MODEL_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_models/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230606.onnx"
MODEL_PATH = "rtmpose-m_wholebody.onnx"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[Init] 軽量AIモデル (約50MB) を自動ダウンロード中...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Init] ダウンロード完了！")

def preprocess(image, input_size=(192, 256)):
    resized = cv2.resize(image, input_size)
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    blob = (blob - mean) / std
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob, image.shape[1], image.shape[0]

def postprocess(simcc_x, simcc_y, orig_w, orig_h, input_size=(192, 256)):
    simcc_x = simcc_x[0]
    simcc_y = simcc_y[0]
    keypoints = []
    scores = []
    for k in range(simcc_x.shape[0]):
        x_idx = np.argmax(simcc_x[k])
        y_idx = np.argmax(simcc_y[k])
        score = float((simcc_x[k, x_idx] + simcc_y[k, y_idx]) / 2.0)
        x = (x_idx / 2.0) * (orig_w / input_size[0])
        y = (y_idx / 2.0) * (orig_h / input_size[1])
        keypoints.append([x, y])
        scores.append(score)
    return np.array(keypoints), np.array(scores)

def extract_biomechanics(pts, conf):
    if len(pts) < 23:
        return None
    com = (pts[11] + pts[12]) / 2.0
    return {
        'com': com,
        'l_heel': pts[19], 'l_toe': pts[17], 'l_conf': (conf[19] + conf[17]) / 2.0,
        'r_heel': pts[22], 'r_toe': pts[20], 'r_conf': (conf[22] + conf[20]) / 2.0
    }

def draw_biomechanics(frame, data):
    h, w, _ = frame.shape
    com = data['com']
    if data['l_conf'] >= data['r_conf']:
        heel, toe = data['l_heel'], data['l_toe']
    else:
        heel, toe = data['r_heel'], data['r_toe']

    heel_x, heel_y = int(heel[0]), int(heel[1])
    toe_x, toe_y = int(toe[0]), int(toe[1])
    com_x, com_y = int(com[0]), int(com[1])

    ground_y = max(heel_y, toe_y)
    bos_min_x = min(heel_x, toe_x)
    bos_max_x = max(heel_x, toe_x)

    # 支持基底面（BOS）
    cv2.line(frame, (bos_min_x, ground_y), (bos_max_x, ground_y), (0, 220, 130), 4)
    cv2.circle(frame, (heel_x, heel_y), 4, (0, 220, 130), -1)
    cv2.circle(frame, (toe_x, toe_y), 4, (0, 220, 130), -1)

    # 垂直ガイド壁
    top_y = int(h * 0.08)
    cv2.line(frame, (bos_min_x, ground_y), (bos_min_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (bos_max_x, ground_y), (bos_max_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)

    # 重心垂下線
    cv2.line(frame, (com_x, com_y), (com_x, ground_y + 20), (238, 211, 34), 2, cv2.LINE_AA)
    cv2.circle(frame, (com_x, com_y), 6, (68, 63, 244), -1)
    cv2.circle(frame, (com_x, ground_y), 4, (238, 211, 34), -1)
    return frame

def run(input_path, output_path):
    ensure_model()
    print("[1/3] ONNX Runtime でモデル起動中...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[2/3] 解析中: {width}x{height} @ {fps:.1f}fps")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        blob, orig_w, orig_h = preprocess(frame)
        outputs = session.run(None, {input_name: blob})
        pts, conf = postprocess(outputs[0], outputs[1], orig_w, orig_h)
        data = extract_biomechanics(pts, conf)
        if data:
            frame = draw_biomechanics(frame, data)
        out.write(frame)
        if count % 30 == 0:
            print(f"  {count} フレーム完了...")

    cap.release()
    out.release()
    print(f"[3/3] 解析完了！ 出力先: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input.mp4")
    parser.add_argument("--output", "-o", default="output.mp4")
    args = parser.parse_args()
    run(args.input, args.output)
EOF
