import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests

# ページのタイトル設定
st.set_page_config(page_title="AI画像分類アプリ", layout="centered")

@st.cache_data
def get_translation_map():
    """ImageNetのIDを日本語に変換する辞書をロード"""
    url = "https://raw.githubusercontent.com/kazunori279/imagenet-japanese/master/imagenet_class_index.json"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        # ID (例: 'n01440764') をキー、日本語名を値とする辞書を作成
        return {v[0]: v[2] for v in data.values()}
    except Exception:
        return {}

@st.cache_resource
def load_model():
    """学習済みモデルをロード（キャッシュして高速化）"""
    return MobileNetV2(weights='imagenet')

def predict(img, model):
    """画像を受け取り、分類結果を返す"""
    # モデルの入力サイズに合わせてリサイズ
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 予測の実行
    preds = model.predict(x)
    # 上位3つの結果をデコード
    return decode_predictions(preds, top=3)[0]

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
        translation_map = get_translation_map()
        results = predict(img, model)
        
    st.subheader("解析結果:")
    for i, (imagenet_id, label, prob) in enumerate(results):
        # 日本語訳を取得（見つからない場合は英語ラベルを使用）
        label_jp = translation_map.get(imagenet_id, label)
        st.write(f"**{i+1}. {label_jp} / {label.replace('_', ' ')}** ({prob*100:.2f}%)")
        st.progress(float(prob))
