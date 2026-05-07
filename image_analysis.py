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
    """ImageNetのインデックスを日本語と英語名に変換する辞書をロード"""
    url = "https://raw.githubusercontent.com/kazunori279/imagenet-japanese/master/imagenet_class_index.json"
    try:
        response = requests.get(url, timeout=5)
        return response.json()
    except Exception:
        return {}

@st.cache_resource
def load_model():
    """学習済みモデルをロード（キャッシュして高速化）"""
    # MobileNetV2の学習済み重みをロード
    model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)
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
        
        # 辞書からラベルを取得（[ID, English, Japanese] の形式）
        labels = class_index.get(idx, ["unknown", "unknown", "不明"])
        label_en = labels[1].replace('_', ' ')
        label_jp = labels[2]
        
        st.write(f"**{i+1}. {label_jp} / {label_en}** ({prob*100:.2f}%)")
        st.progress(prob)
