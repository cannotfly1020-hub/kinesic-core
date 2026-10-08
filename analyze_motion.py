import argparse
import os
import urllib.request
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime が見つかりません。ターミナルで『pip install onnxruntime opencv-python』を実行してください。")

# ==============================================================================
# RTMPose-m (COCO-WholeBody: 133キーポイント) 公式ONNXモデル
# 足部詳細キーポイント:
# 11: 左股関節, 12: 右股関節
# 17: 左母趾先端, 18: 左小趾先端, 19: 左踵
# 20: 右母趾先端, 21: 右小趾先端, 22: 右踵
# ==============================================================================
MODEL_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_models/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230606.onnx"
MODEL_PATH = "rtmpose-m_wholebody.onnx"

def ensure_model():
    """モデルファイルが存在しない場合は自動ダウンロード (約50MB)"""
    if not os.path.exists(MODEL_PATH):
        print("[Init] RTMPose WholeBody (133点) モデルをダウンロード中 (約50MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Init] ダウンロード完了！")

def preprocess(image, input_size=(192, 256)):
    """画像をRTMPoseの入力形式 (256x192) に変換・正規化"""
    h, w, _ = image.shape
    resized = cv2.resize(image, input_size)
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    # ImageNet 正規化
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    blob = (blob - mean) / std
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)  # NCHW
    return blob, w, h

def postprocess(simcc_x, simcc_y, orig_w, orig_h, input_size=(192, 256)):
    """SimCC形式の出力から133点の座標と信頼度スコアを復元"""
    simcc_x = simcc_x[0]  # [133, 384]
    simcc_y = simcc_y[0]  # [133, 512]

    keypoints = []
    scores = []
    for k in range(simcc_x.shape[0]):
        x_idx = np.argmax(simcc_x[k])
        y_idx = np.argmax(simcc_y[k])
        x_score = simcc_x[k, x_idx]
        y_score = simcc_y[k, y_idx]
        score = float((x_score + y_score) / 2.0)

        # 0.5ピクセル刻みのSimCC表現を元動画解像度へスケーリング
        x = (x_idx / 2.0) * (orig_w / input_size[0])
        y = (y_idx / 2.0) * (orig_h / input_size[1])

        keypoints.append([x, y])
        scores.append(score)

    return np.array(keypoints), np.array(scores)

def extract_biomechanics(pts, conf):
    """
    COCO-WholeBody キーポイントから力学要素を抽出:
    - 身体重心 (COM): 左右股関節 (11, 12) の中点
    - 足部: 踵(19, 22), 母趾(17, 20), 小趾(18, 21)
    """
    if len(pts) < 23:
        return None

    # 重心 (COM)
    com = (pts[11] + pts[12]) / 2.0

    # 左足 (踵, 母趾, 小趾)
    l_heel = pts[19]
    l_big_toe = pts[17]
    l_small_toe = pts[18]
    l_conf = (conf[19] + conf[17] + conf[18]) / 3.0

    # 右足 (踵, 母趾, 小趾)
    r_heel = pts[22]
    r_big_toe = pts[20]
    r_small_toe = pts[21]
    r_conf = (conf[22] + conf[20] + conf[21]) / 3.0

    return {
        'com': com,
        'l_heel': l_heel, 'l_big_toe': l_big_toe, 'l_small_toe': l_small_toe, 'l_conf': l_conf,
        'r_heel': r_heel, 'r_big_toe': r_big_toe, 'r_small_toe': r_small_toe, 'r_conf': r_conf
    }

def draw_biomechanics(frame, data):
    """BOS (支持基底面) と重心垂下線の描画"""
    h, w, _ = frame.shape
    com = data['com']

    # 信頼度の高い側の足を優先（立脚側）
    if data['l_conf'] >= data['r_conf']:
        heel = data['l_heel']
        toe = data['l_big_toe']
    else:
        heel = data['r_heel']
        toe = data['r_big_toe']

    heel_x, heel_y = int(heel[0]), int(heel[1])
    toe_x, toe_y = int(toe[0]), int(toe[1])
    com_x, com_y = int(com[0]), int(com[1])

    # 床面高さ
    ground_y = max(heel_y, toe_y)
    bos_min_x = min(heel_x, toe_x)
    bos_max_x = max(heel_x, toe_x)

    # 1. 支持基底面 (BOS) の接地面バー (エメラルドグリーン)
    cv2.line(frame, (bos_min_x, ground_y), (bos_max_x, ground_y), (0, 220, 130), 4)
    cv2.circle(frame, (heel_x, heel_y), 5, (0, 220, 130), -1)
    cv2.circle(frame, (toe_x, toe_y), 5, (0, 220, 130), -1)

    # 2. 支持基底面の端から真上へ伸びる垂直壁ガイド (上向きの壁)
    top_y = int(h * 0.08)
    cv2.line(frame, (bos_min_x, ground_y), (bos_min_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (bos_max_x, ground_y), (bos_max_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)

    # 3. 身体重心 (COM) から真下に落ちる重力垂下線 (水色)
    cv2.line(frame, (com_x, com_y), (com_x, ground_y + 20), (238, 211, 34), 2, cv2.LINE_AA)
    # 重心ポイント (赤)
    cv2.circle(frame, (com_x, com_y), 7, (68, 63, 244), -1)
    cv2.circle(frame, (com_x, com_y), 8, (255, 255, 255), 1)
    # 床面落下点
    cv2.circle(frame, (com_x, ground_y), 4, (238, 211, 34), -1)

    return frame

def run(input_path, output_path):
    ensure_model()

    print("[1/3] ONNX Runtime で RTMPose (COCO-WholeBody) を起動中...")
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

    print(f"[2/3] RTMPose 解析中: {width}x{height} @ {fps:.1f}fps")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        blob, orig_w, orig_h = preprocess(frame)
        outputs = session.run(None, {input_name: blob})
        simcc_x, simcc_y = outputs[0], outputs[1]

        pts, conf = postprocess(simcc_x, simcc_y, orig_w, orig_h)
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
    parser.add_argument("--input", "-i", default="input.mp4", help="入力動画パス")
    parser.add_argument("--output", "-o", default="output.mp4", help="出力動画パス")
    args = parser.parse_args()
    run(args.input, args.output)
