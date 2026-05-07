import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import requests

# ページのタイトル設定
st.set_page_config(page_title="AI画像分類アプリ", layout="centered")

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

    with torch.no_grad():
        output = model(input_batch)
    
    # ソフトマックス関数で確率に変換し、上位3つを取得
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    return top3_prob.tolist(), top3_indices.tolist()

# UI部分
st.title("🖼️ 画像オブジェクト分類アプリ")
st.write("画像をアップロードすると、AIが何が写っているかを解析します。")

uploaded_file = st.file_uploader("画像ファイルを選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の表示
    img = Image.open(uploaded_file)
    st.image(img, caption="アップロードされた画像", use_container_width=True)
    
    with st.spinner('解析中...'):
        # モデルのロードと予測
        model = load_model()
        class_index = get_translation_map()
        probs, indices = predict(img, model)
        
    st.subheader("解析結果:")
    for i in range(len(indices)):
        idx = str(indices[i])
        prob = probs[i]
        
        # 辞書からラベルデータを取得
        label_data = class_index.get(idx, {"en": "unknown", "jp": None})
        label_en = label_data["en"].replace('_', ' ')
        label_jp = label_data["jp"]
        
        if label_jp:
            st.write(f"**{i+1}. {label_jp} / {label_en}** ({prob*100:.2f}%)")
        else:
            st.write(f"**{i+1}. {label_en}** ({prob*100:.2f}%)")
        st.progress(prob)
