import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import requests
import io
import os
import hashlib # Added for password hashing

# ページのタイトル設定
st.set_page_config(page_title="AI画像分類アプリ", layout="centered")

# --- 認証機能 ---
# ハードコードされた認証情報（デモンストレーション用）
# 実際のアプリケーションでは、より安全な方法で管理してください。
USERNAME = "cq"
# パスワードをハッシュ化して保存
PASSWORD_HASH = hashlib.sha256("cq001".encode()).hexdigest()

def check_password(username, password):
    """入力されたユーザー名とパスワードが正しいかチェックする"""
    if username == USERNAME and hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
        return True
    return False

# 認証状態をセッションステートで管理
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("ログイン")
    with st.form("login_form"):
        username_input = st.text_input("ユーザー名")
        password_input = st.text_input("パスワード", type="password")
        login_button = st.form_submit_button("ログイン")

        if login_button:
            if check_password(username_input, password_input):
                st.session_state.authenticated = True
                st.rerun() # ログイン成功後、アプリを再実行してメインコンテンツを表示
            else:
                st.error("ユーザー名またはパスワードが間違っています。")
else:
    # --- 認証成功後のメインアプリケーションコンテンツ ---
    # サイドバー設定
    st.sidebar.header("機能設定")
    enable_cat_crop = st.sidebar.checkbox("猫の自動切り抜きを有効にする", value=False, help="ONにすると物体検出モデルをロードし、猫を抽出します。")

    # ログアウトボタン
    if st.sidebar.button("ログアウト"):
        st.session_state.authenticated = False
        st.rerun() # ログアウト後、アプリを再実行してログイン画面に戻る

@st.cache_data
def get_translation_map():
    """ImageNetのインデックスを日本語と英語名に変換する辞書をロード。失敗時は英語のみを返す。"""
    # torchvisionのモデルメタデータから標準の英語ラベルを取得
    weights = models.MobileNet_V2_Weights.DEFAULT
    en_labels = weights.meta["categories"]
    
    url = "https://raw.githubusercontent.com/kazunori279/imagenet-japanese/master/imagenet_class_index.json"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status() # HTTPエラーが発生した場合に例外を発生させる
        jp_data = response.json()
        # インデックスをキーに、英語と日本語のペアを辞書化
        return {k: {"en": en_labels[int(k)], "jp": v[2]} for k, v in jp_data.items()}
    except Exception:
        # 日本語データの取得に失敗した場合は、日本語をNoneとして英語ラベルのみを返す
        return {str(i): {"en": label, "jp": None} for i, label in enumerate(en_labels)}

@st.cache_resource
def load_model():
    """学習済みモデルをロード（キャッシュして高速化）"""
    # MobileNetV2の学習済み重みをロード
    model = models.mobilenet_v2(weights="DEFAULT")
    model.eval()  # 推論モードに設定
    return model

@st.cache_resource
def load_detection_model():
    """物体検出モデル（Faster R-CNN）をロード"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    return model, weights.meta["categories"]

@torch.inference_mode()
def detect_and_crop_cat(img, det_model, categories, threshold=0.8):
    """猫を検出し、最も確信度の高い個体をクロップする"""
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img)
    
    prediction = det_model([input_tensor])[0]
    
    # COCOデータセットで猫のラベルは 'cat'
    cat_label_idx = [i for i, cat in enumerate(categories) if cat == 'cat'][0]
    
    # スコアが高い順に猫を探す
    best_box = None
    for i, label in enumerate(prediction['labels']):
        if label == cat_label_idx and prediction['scores'][i] > threshold:
            best_box = prediction['boxes'][i].tolist()
            break
    
    if best_box:
        # クロップ実行 [xmin, ymin, xmax, ymax]
        cropped_img = img.crop((best_box[0], best_box[1], best_box[2], best_box[3]))
        return cropped_img
    return None

def get_download_data(img, format="PNG"):
    """画像をバイトデータに変換"""
    buf = io.BytesIO()
    # RGBAならRGBに変換して保存（JPEG対応等のため）
    save_img = img.convert("RGB") if img.mode == "RGBA" else img
    save_img.save(buf, format=format)
    return buf.getvalue()

@torch.inference_mode()
def predict(img, model):
    """画像を受け取り、分類結果を返す"""
    # PyTorch用の前処理（リサイズ、テンソル化、正規化）
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # バッチ次元の追加

    output = model(input_batch)
    
    # ソフトマックス関数で確率に変換し、上位3つを取得
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    return top3_prob.tolist(), top3_indices.tolist()

if st.session_state.authenticated:
    # UI部分
    st.title("🖼️ 画像オブジェクト分類アプリ")
    st.write("画像をアップロードすると、AIが何が写っているかを解析します。")

    # 複数ファイルのアップロードを許可
    uploaded_files = st.file_uploader("画像ファイルを選択してください...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files: # 複数のファイルがアップロードされた場合
        # モデルと翻訳マップは一度だけロード（@st.cache_resource/@st.cache_dataにより）
        model = load_model()
        class_index = get_translation_map()
        
        # 猫検知が有効な場合のみ物体検出モデルをロード
        det_model, categories = None, None
        if enable_cat_crop:
            det_model, categories = load_detection_model()
        
        # 2列のグリッドレイアウトを作成して垂直方向のスクロールを削減
        cols = st.columns(2)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # インデックスに応じて左右のカラムに振り分け
            with cols[i % 2]:
                with st.container(border=True):
                    # 画像の読み込み
                    img = Image.open(uploaded_file)
                    
                    # 小さなプレビュー画像を表示
                    st.image(img, use_container_width=True)
                    # クリックして拡大表示するためのポップオーバー
                    with st.popover("🔍 画像を拡大して確認"):
                        st.image(img, caption=uploaded_file.name, use_container_width=True)
                    
                    st.write(f"**{uploaded_file.name}**")
                    
                    with st.spinner('解析中...'):
                        probs, indices = predict(img, model)
                    
                    # 猫が含まれているか判定（ImageNetの猫関連ID: 281-285）
                    is_cat = any(281 <= idx <= 285 for idx in indices)
                    
                    if is_cat and enable_cat_crop:
                        st.success("🐱 猫を検知しました！切り抜きを作成します。")
                        cropped_cat = detect_and_crop_cat(img, det_model, categories)
                        if cropped_cat:
                            st.image(cropped_cat, caption="切り抜かれた猫", width=150)
                            
                            # ファイル名の生成
                            base_name, ext = os.path.splitext(uploaded_file.name)
                            cat_file_name = f"{base_name}_CAT{ext}"
                            
                            # ダウンロードボタン
                            btn_data = get_download_data(cropped_cat, format="PNG" if ext.lower()==".png" else "JPEG")
                            st.download_button(label="💾 クロップ画像を保存", data=btn_data, file_name=cat_file_name, mime=f"image/{ext[1:]}")

                    for j in range(len(indices)):
                        idx = str(indices[j])
                        prob = probs[j]
                        label_data = class_index.get(idx, {"en": "unknown", "jp": None})
                        label_en = label_data["en"].replace('_', ' ')
                        label_jp = label_data["jp"]
                        
                        display_name = f"{label_jp} / {label_en}" if label_jp else label_en
                        st.write(f"{j+1}. {display_name} ({prob*100:.1f}%)")
                        st.progress(prob)
