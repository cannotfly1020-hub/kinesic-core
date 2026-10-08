import argparse
import cv2
import numpy as np

try:
    from rtmlib import Wholebody, draw_skeleton
except ImportError:
    raise ImportError("rtmlib が見つかりません。ターミナルで『pip install rtmlib』を実行してください。")

def extract_biomechanics(keypoints, scores):
    """
    COCO-WholeBody (133点) から力学要素を抽出:
    11: 左股関節, 12: 右股関節
    17: 左母趾先端, 19: 左踵
    20: 右母趾先端, 22: 右踵
    """
    if len(keypoints) == 0:
        return None

    # 最前面の人物（0人目）を対象
    pts = keypoints[0]
    conf = scores[0]

    if len(pts) < 23:
        return None

    # 重心 (COM): 左右股関節中点
    com = (pts[11] + pts[12]) / 2.0

    # 左足 (踵, 母趾先端)
    l_heel = pts[19]
    l_toe = pts[17]
    l_conf = (conf[19] + conf[17]) / 2.0

    # 右足 (踵, 母趾先端)
    r_heel = pts[22]
    r_toe = pts[20]
    r_conf = (conf[22] + conf[20]) / 2.0

    return {
        'com': com,
        'l_heel': l_heel, 'l_toe': l_toe, 'l_conf': l_conf,
        'r_heel': r_heel, 'r_toe': r_toe, 'r_conf': r_conf
    }

def draw_biomechanics(frame, data):
    """支持基底面 (BOS) と 重心垂下線 (COM) を描画"""
    h, w, _ = frame.shape
    com = data['com']

    # 信頼度の高い立脚側を選択
    if data['l_conf'] >= data['r_conf']:
        heel = data['l_heel']
        toe = data['l_toe']
    else:
        heel = data['r_heel']
        toe = data['r_toe']

    heel_x, heel_y = int(heel[0]), int(heel[1])
    toe_x, toe_y = int(toe[0]), int(toe[1])
    com_x, com_y = int(com[0]), int(com[1])

    ground_y = max(heel_y, toe_y)
    bos_min_x = min(heel_x, toe_x)
    bos_max_x = max(heel_x, toe_x)

    # 1. 支持基底面 (BOS) 接地面バー (エメラルドグリーン)
    cv2.line(frame, (bos_min_x, ground_y), (bos_max_x, ground_y), (0, 220, 130), 4)
    cv2.circle(frame, (heel_x, heel_y), 5, (0, 220, 130), -1)
    cv2.circle(frame, (toe_x, toe_y), 5, (0, 220, 130), -1)

    # 2. 支持基底面の端から真上へ伸びる垂直壁ガイド
    top_y = int(h * 0.08)
    cv2.line(frame, (bos_min_x, ground_y), (bos_min_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (bos_max_x, ground_y), (bos_max_x, top_y), (0, 180, 100), 1, cv2.LINE_AA)

    # 3. 身体重心 (COM) 鉛直垂下線 (水色)
    cv2.line(frame, (com_x, com_y), (com_x, ground_y + 20), (238, 211, 34), 2, cv2.LINE_AA)
    cv2.circle(frame, (com_x, com_y), 7, (68, 63, 244), -1)
    cv2.circle(frame, (com_x, com_y), 8, (255, 255, 255), 1)
    cv2.circle(frame, (com_x, ground_y), 4, (238, 211, 34), -1)

    return frame

def run(input_path, output_path):
    print("[1/3] RTMPose WholeBody (133点) を起動中...")
    # balancedモード: RTMPose-m (Wholebody 133点) をONNXで自動取得・実行
    wholebody = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[2/3] 解析開始: {width}x{height} @ {fps:.1f}fps")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        # RTMPose WholeBody 推論 (人物検出 + 133点姿勢推定)
        keypoints, scores = wholebody(frame)

        data = extract_biomechanics(keypoints, scores)
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
