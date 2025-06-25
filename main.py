import streamlit as st
import pycountry
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim


# ===== 工具函式 =====
def get_flag_image(country_name):
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        code = country.alpha_2.lower()
        url = f"https://flagcdn.com/w320/{code}.png"
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception:
        return None


def analyze_flag_color(img, n_colors=5, resize_to=(100, 100)):
    img = img.resize(resize_to)
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    centers = kmeans.cluster_centers_
    proportions = counts / counts.sum()
    sorted_idx = np.argsort(-proportions)
    return centers[sorted_idx], proportions[sorted_idx]


def blend_colors(colors, proportions):
    return np.dot(proportions, colors).astype(np.uint8)


def render_color_bar(colors, proportions):
    bar = np.zeros((50, 300, 3), dtype=np.uint8)
    start = 0
    for color, prop in zip(colors.astype(np.uint8), proportions):
        end = start + int(prop * 300)
        bar[:, start:end, :] = color
        start = end
    return Image.fromarray(bar)


def render_blended_color(blended_color):
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:, :, :] = blended_color
    return Image.fromarray(img)


def get_country_center_latlon(country_name):
    try:
        geolocator = Nominatim(user_agent="flag-locator")
        location = geolocator.geocode(country_name)
        if location:
            return (location.latitude, location.longitude)
    except:
        return None
    return None


def render_country_map(country_name):
    coords = get_country_center_latlon(country_name)
    if coords:
        m = folium.Map(location=coords, zoom_start=4)
        folium.Marker(coords, tooltip=country_name).add_to(m)
        return m
    else:
        return None


# ===== Streamlit 主程式 =====
st.set_page_config("國旗融合視覺化", layout="wide")
st.title("🌐 國旗融合色彩分析（世界定位版）")

country_input = st.text_input("請輸入國家英文名稱（如 France, Japan, Brazil）")

if country_input:
    img = get_flag_image(country_input)
    if img:
        colors, proportions = analyze_flag_color(img)
        blended_color = blend_colors(colors, proportions)

        # 建立四格版面
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始國旗")
            st.image(img, width=300)
        with col2:
            st.subheader("顏色比例")
            st.image(render_color_bar(colors, proportions))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("世界地圖位置")
            map_obj = render_country_map(country_input)
            if map_obj:
                st_folium(map_obj, width=500, height=300)
            else:
                st.error("找不到該國家的地圖位置")

        with col4:
            st.subheader("融合代表色")
            st.image(render_blended_color(blended_color), caption=f"RGB: {tuple(blended_color)}")
    else:
        st.error("❌ 找不到國旗圖，請檢查拼字")
