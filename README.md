# hand-control

Webカメラ映像から手の動きを検知し、ジェスチャーで画面を操作するプロジェクト。
MediaPipe Tasks API + OpenCV + pyautogui を使用。

## セットアップ

### 必要環境

- Python 3.12(MediaPipeの対応バージョンに合わせて固定)
- Webカメラ
- [uv](https://docs.astral.sh/uv/)

### インストール

```powershell
uv python pin 3.12
uv add mediapipe opencv-python numpy pyautogui
```


### モデルファイルのダウンロード

使うスクリプトに応じて、対応する `.task` / `.tflite` ファイルをプロジェクト直下(`src/my_app/`)に置く。

| ファイル | 用途 | ダウンロードURL |
|---|---|---|
| `hand_landmarker.task` | 手のランドマーク検出 | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |


PowerShellでのダウンロード例:

```powershell
Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -OutFile hand_landmarker.task
```

## スクリプト一覧

| ファイル | 内容 |
|---|---|
| `hand_screen_control.py` | 手のジェスチャーだけで画面を実際に操作する |

`uv run python src/hand_control/hand_screen_control.py` | `uv run hand-control` で実行。カメラ映像ウィンドウで `q` または `ESC` を押すと終了する。

## hand_screen_control.py の操作方法

顔検出は使わず、片手のトラッキングに全振りして精度・速度を優先した、画面操作用のメインスクリプト。

| ジェスチャー | 動作 |
|---|---|
| 人差し指を動かす(パーの状態) | マウスカーソル移動 |
| パー → グー | ドラッグ開始(グーのまま移動でドラッグ継続) |
| グー → パー | ドロップ(ドラッグ終了) |
| 親指と人差し指をつまんで、すぐ離す | クリック |
| つまんだまま少しキープしてから離す | ダブルクリック |
| パーのまま左右に振る | スライド送り(→/←キー) |
| ピースサイン | 音量ミュート切り替え |

### 主な調整パラメータ(スクリプト冒頭)

- `MIN_DETECTION_CONFIDENCE` / `MIN_TRACKING_CONFIDENCE`: 検出の厳しさ。誤検知が多いなら上げる、反応が悪いなら下げる
- `SMOOTHING`: カーソルの滑らかさ(0〜1、大きいほど滑らかだが遅延も増える)
- `PINCH_ENTER_RATIO` / `PINCH_EXIT_RATIO`: ピンチ(つまみ)判定のしきい値。ヒステリシスを持たせて誤反応を防止
- `LONG_PINCH_SECONDS`: これ以上つまみをキープしてから離すとダブルクリック扱い
- `FRAME_MARGIN`: カメラ映像のうち、操作エリアとして使う中央部分の割合


## 既知の注意点

- `pyautogui`にはフェイルセーフ機能があり、カーソルが画面の左上端(0,0)に達すると強制的に停止する(暴走時の緊急停止用)
- Webカメラへのアクセス権限が必要(Windowsの「設定 > プライバシー > カメラ」で許可)