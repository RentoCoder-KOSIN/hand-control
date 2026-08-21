"""
hand_screen_control.py
--------------------------
顔検出は使わず、片手のトラッキングだけに全振りして画面を操作する。

できること:
    1) 人差し指の先でマウスカーソルを移動
    2) パー→グー でドラッグ開始、グー→パー でドロップ
    3) 親指と人差し指をつまんで すぐ離す → クリック
       つまんだまま少しキープしてから離す → ダブルクリック
    4) パーを出したまま左右に振る → スライド送り(左右矢印キー)
    5) ピースサイン → 音量ミュート切り替え

必要なライブラリ:
    uv add mediapipe opencv-python numpy pyautogui

必要なモデルファイル:
    hand_landmarker.task (前回ダウンロード済みならそのまま流用可)
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

使い方:
    python hand_screen_control.py
    カメラ映像ウィンドウで 'q' または ESC で終了

注意:
    - pyautoguiにはフェイルセーフ機能があり、カーソルを画面の左上端(0,0)に
      移動させると強制的に例外を投げて安全停止できます(暴走時の緊急停止用)。
    - 実際に画面操作を始めると、他の操作がしにくくなるので、
      慣れるまではカメラ映像ウィンドウを画面の隅に置いて動作確認しながら使うのがおすすめです。
"""

import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ==== 設定項目 ====
CAMERA_INDEX = 0
MODEL_PATH = "hand_landmarker.task"

# カメラ解像度(上げるほど指先の位置精度が上がるが、処理は重くなる)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 検出の厳しさ(誤検知を減らしたいなら上げる。反応が悪いなら下げる)
MIN_DETECTION_CONFIDENCE = 0.9
MIN_TRACKING_CONFIDENCE = 0.9

# カーソル操作まわり
SMOOTHING = 0.8          # 0〜1。大きいほど滑らかだが反応が遅れる
FRAME_MARGIN = 0.15      # カメラ映像の外側何%を「操作エリア外」として無視するか

# 親指-人差し指の距離(手のサイズに対する比率)がこれを下回ったら「つまんだ」と判定。
# ENTER(つかみ始め)の方を厳しく、EXIT(離す)の方を緩くすることで、
# 閾値ギリギリでのチカチカ(誤ON/OFF)を防ぐ(ヒステリシス)。
PINCH_ENTER_RATIO = 0.23
PINCH_EXIT_RATIO = 0.26

# ピンチをこの秒数以上キープしてから離すと「ダブルクリック」、それ未満なら「クリック」。
LONG_PINCH_SECONDS = 0.4

# ジェスチャーショートカットまわり
SWIPE_COOLDOWN_SECONDS = 0.5
SWIPE_DISTANCE_THRESHOLD = 0.15  # 手首のx移動量(正規化座標)がこれを超えたらスワイプ判定
PEACE_COOLDOWN_SECONDS = 1.0

WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 2, 5, 9, 13, 17

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # moveTo等のたびに勝手に待たされるのを防ぐ(こちらでフレームレート制御する)

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


def _distance(a, b):
    """3D距離(x, y, z)。zも含めることで、2D画面上でたまたま指が重なって
    見えているだけ(奥行きは離れている)のケースを誤ってピンチと判定しないようにする"""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def get_finger_states(landmarks):
    """
    各指(親指・人差し指・中指・薬指・小指)が伸びているかを
    手首からの距離ベースで判定し、[bool, bool, bool, bool, bool] で返す
    """
    wrist = landmarks[WRIST]
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    states = []
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_dist = _distance(landmarks[tip_idx], wrist)
        mcp_dist = _distance(landmarks[mcp_idx], wrist)
        states.append(tip_dist > mcp_dist * 1.3)
    return states


def get_pinch_ratio(landmarks):
    """親指-人差し指の距離を、手のサイズ(手首-中指付け根)で正規化した比率を返す"""
    hand_size = _distance(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if hand_size == 0:
        return 999.0
    pinch_dist = _distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    return pinch_dist / hand_size


def map_to_screen(x, y):
    """
    カメラ映像内の正規化座標(0〜1)を、FRAME_MARGIN分だけ中央に寄せたうえで
    画面全体の座標にマッピングする(手を画面端まで伸ばさなくても端まで届くように)
    """
    x = np.clip((x - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN), 0, 1)
    y = np.clip((y - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN), 0, 1)
    return x * SCREEN_WIDTH, y * SCREEN_HEIGHT


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,  # 操作用の手は1本に絞って精度・速度を優先
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        print("カメラを開けませんでした。CAMERA_INDEXを確認してください。")
        return

    smoothed_x, smoothed_y = None, None
    pending_grab = False   # パーだった実績があり、まだグーで確定していない(中間状態も含む)
    is_dragging = False
    drag_offset = (0.0, 0.0)  # ドラッグ開始時の「人差し指カーソル位置」と「手のひら中心位置」のズレ

    pinch_active = False
    pinch_start_time = None

    last_wrist_x = None
    last_swipe_time = 0.0
    last_peace_time = 0.0

    def finish_pinch():
        """ピンチが終了した瞬間に呼ぶ。キープ時間に応じてクリック/ダブルクリックを発行する"""
        nonlocal pinch_active, pinch_start_time
        if pinch_start_time is not None:
            duration = time.time() - pinch_start_time
            if duration >= LONG_PINCH_SECONDS:
                pyautogui.doubleClick()
            else:
                pyautogui.click()
        pinch_active = False
        pinch_start_time = None

    print("ハンドコントロールを開始します。'q' または ESC で終了。")
    print(f"画面解像度: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                h, w, _ = frame.shape

                finger_states = get_finger_states(landmarks)
                # 親指(0番目)は角度のブレが大きいので、パー/グー判定は
                # 人差し指・中指・薬指・小指(1〜4番目)だけで見る
                four_fingers = finger_states[1:5]
                is_open = all(four_fingers)
                is_fist = not any(four_fingers)

                palm_x, palm_y = map_to_screen(
                    landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y
                )

                # --- 1) パー→グー でドラッグ開始 / グー→パー でドロップ ---
                # 「パー」「ドラッグ確定後のグー」以外の中間的な手の形の間は、
                # カーソル座標(target_x/y)を更新しない = カーソルを凍結する。
                # こうしないと、指を閉じていく途中の物理的な動きにカーソルが
                # つられてズレてしまう。
                target_x, target_y = smoothed_x, smoothed_y  # デフォルトは「凍結」(前回位置のまま)

                if is_open:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                    pending_grab = True
                    index_tip = landmarks[INDEX_TIP]
                    target_x, target_y = map_to_screen(index_tip.x, index_tip.y)

                elif is_fist:
                    if pending_grab and not is_dragging:
                        # つかんだ瞬間:今のカーソル位置(人差し指基準)と
                        # 手のひら中心位置とのズレを記録しておく(切り替え時のワープ防止)
                        drag_offset = (smoothed_x - palm_x, smoothed_y - palm_y) \
                            if smoothed_x is not None else (0.0, 0.0)
                        pyautogui.mouseDown()
                        is_dragging = True
                    pending_grab = False
                    if is_dragging:
                        target_x = palm_x + drag_offset[0]
                        target_y = palm_y + drag_offset[1]
                    # is_dragging が False(パーを経ずに突然グーにした等)ならカーソルは凍結のまま

                else:
                    # 開ききっても閉じきってもいない中間状態
                    if is_dragging:
                        # ドラッグ中に多少グーが緩んでも(検出のブレ)、ドラッグ自体は継続する
                        target_x = palm_x + drag_offset[0]
                        target_y = palm_y + drag_offset[1]
                    # ドラッグ中でなければ pending_grab は維持したままカーソルは凍結

                # --- 2) カーソル移動(凍結時は target が smoothed と同じなので実質何もしない) ---
                if smoothed_x is None:
                    smoothed_x, smoothed_y = target_x, target_y
                else:
                    smoothed_x = smoothed_x * SMOOTHING + target_x * (1 - SMOOTHING)
                    smoothed_y = smoothed_y * SMOOTHING + target_y * (1 - SMOOTHING)

                try:
                    pyautogui.moveTo(smoothed_x, smoothed_y)
                except pyautogui.FailSafeException:
                    print("フェイルセーフが作動しました(カーソルが画面隅へ)。停止します。")
                    break

                # --- 3) ピンチでクリック/ダブルクリック ---
                pinch_ratio = get_pinch_ratio(landmarks)
                if not pinch_active and pinch_ratio < PINCH_ENTER_RATIO:
                    pinch_active = True
                    pinch_start_time = time.time()
                elif pinch_active and pinch_ratio > PINCH_EXIT_RATIO:
                    finish_pinch()

                now = time.time()

                # --- 4) パー(全部伸びている)を左右に振ったらスライド送り ---
                if is_open:
                    wrist_x = landmarks[WRIST].x
                    if last_wrist_x is not None and now - last_swipe_time > SWIPE_COOLDOWN_SECONDS:
                        delta = wrist_x - last_wrist_x
                        if delta > SWIPE_DISTANCE_THRESHOLD:
                            pyautogui.press("right")
                            print("スワイプ→: 次のスライド")
                            last_swipe_time = now
                        elif delta < -SWIPE_DISTANCE_THRESHOLD:
                            pyautogui.press("left")
                            print("スワイプ←: 前のスライド")
                            last_swipe_time = now
                    last_wrist_x = wrist_x
                else:
                    last_wrist_x = None

                # --- 5) ピース(人差し指+中指だけ)で音量ミュート切り替え ---
                if finger_states == [False, True, True, False, False]:
                    if now - last_peace_time > PEACE_COOLDOWN_SECONDS:
                        pyautogui.press("volumemute")
                        print("ピース: ミュート切り替え")
                        last_peace_time = now

                # --- デバッグ用オーバーレイ ---
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)
                status = "DRAGGING" if is_dragging else ("PINCH" if pinch_active else "")
                cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"pinch_ratio: {pinch_ratio:.2f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                smoothed_x, smoothed_y = None, None
                pending_grab = False
                # 手の検出が一瞬途切れても、進行中だった操作はきちんと後始末する
                if pinch_active:
                    finish_pinch()
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

            cv2.imshow("Hand Control (camera)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        if is_dragging:
            pyautogui.mouseUp()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()