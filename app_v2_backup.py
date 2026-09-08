import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from PIL import Image
import requests
import io

# ==========================================
# 0. 페이지 및 깃허브 설정
# ==========================================
st.set_page_config(page_title="폰트 클러스터링 통합 대시보드", layout="wide")

# Streamlit Secrets에서 GitHub 정보 가져오기
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = st.secrets["REPO_OWNER"] # 깃허브 아이디 (예: hyowon-1592)
    REPO_NAME = st.secrets["REPO_NAME"]   # 레포지토리 이름 (예: dashboard_clustering)
    BRANCH = st.secrets.get("BRANCH", "main")
except KeyError:
    st.error("Streamlit Secrets 설정이 누락되었습니다. 깃허브 토큰과 레포지토리 정보를 설정해주세요.")
    st.stop()

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.raw"
}

# 💡 수정된 부분: 데이터가 들어있는 최상위 폴더 경로 지정
DATA_ROOT = "data_clustering_v2"

# ==========================================
# 1. 공통 헬퍼 함수 (GitHub에서 파일 읽어오기)
# ==========================================
@st.cache_data(show_spinner=False)
def get_image_from_github(folder_name, file_name_without_ext):
    """GitHub Private Repo에서 이미지를 다운로드하여 PIL Image로 반환"""
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        # DATA_ROOT 경로 추가
        file_path = f"{DATA_ROOT}/{folder_name}/{file_name_without_ext}{ext}"
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{file_path}"
        
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    return None

def get_text_from_github(file_path):
    """GitHub Private Repo에서 텍스트 파일을 읽어오기"""
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{DATA_ROOT}/{file_path}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.text
    else:
        # 실패 이유를 화면에 직접 출력합니다 (원인 파악용)
        st.error(f"요청 URL: {url}")
        st.error(f"실패 상태 코드: {response.status_code}")
        st.error(f"깃허브 응답 내용: {response.text}")
        return None

# ==========================================
# 2. C & G 폰트 공통 데이터 로드 함수
# ==========================================
@st.cache_data
def load_data_2D(file_path):
    text_data = get_text_from_github(file_path)
    if not text_data: return None, None
    
    blocks = text_data.split("파일명:")[1:]
    data_list = []
    
    for block in blocks:
        filename = block.split('\n')[0].strip()
        top_match = re.search(r"상단 대비\s*([\d\.]+)\s*배", block)
        bottom_match = re.search(r"하단 대비\s*([\d\.]+)\s*배", block)
        
        if top_match and bottom_match:
            top_val, bottom_val = float(top_match.group(1)), float(bottom_match.group(1))
            if top_val < 5.0 and bottom_val < 5.0:
                data_list.append({"filename": filename, "top_ratio": top_val, "bottom_ratio": bottom_val})
                
    if not data_list: return None, None
    df = pd.DataFrame(data_list)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['top_ratio', 'bottom_ratio']].values).astype(str)
    return df, kmeans.cluster_centers_

# ==========================================
# 3. 각 폰트별 페이지 렌더링 함수
# ==========================================
def show_CG_page(font_name):
    st.title(f"{font_name} 폰트 상단/하단 두께 비율 클러스터링")
    
    result_folder = f"{font_name}_result"
    file_path = f"{result_folder}/{font_name}_basic_analysis.txt"
    
    df, centroids = load_data_2D(file_path)
    if df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(df, x="top_ratio", y="bottom_ratio", color="cluster", hover_data=["filename"],
                             labels={"top_ratio": "상단 대비 비율 (배)", "bottom_ratio": "하단 대비 비율 (배)", "cluster": "그룹"})
            fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                                     marker=dict(color='red', symbol='x', size=12), name='중심점', hoverinfo='skip'))
            fig.update_layout(height=600)
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
            
        with col2:
            st.subheader("선택된 이미지 확인")
            if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
                selected_filename = event.selection.points[0]["customdata"][0]
                orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
                
                st.markdown(f"**원본 파일:** `{orig_fname}`")
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: st.image(orig_img, use_container_width=True)
                else: st.warning("원본 이미지를 찾을 수 없습니다.")
                
                st.divider()
                
                actual_crop = f"{selected_filename}_{font_name}_stroke_basic"
                st.markdown(f"**크롭 파일:** `{actual_crop}`")
                crop_img = get_image_from_github(result_folder, actual_crop)
                if crop_img: st.image(crop_img, use_container_width=True)
                else: st.warning("크롭 이미지를 찾을 수 없습니다.")
    else:
        st.error(f"'{file_path}' 데이터를 찾을 수 없거나 내용이 비어있습니다.")

@st.cache_data
def load_data_U(file_path):
    text_data = get_text_from_github(file_path)
    if not text_data: return None, None
    
    blocks = text_data.split("파일명:")[1:]
    data_list = []
    
    for block in blocks:
        filename = block.split('\n')[0].strip()
        ratio_match = re.search(r"왼쪽 기둥이 오른쪽 기둥 대비\s*([\d\.]+)\s*배", block)
        if ratio_match and float(ratio_match.group(1)) < 5.0:
            data_list.append({"filename": filename, "ratio": float(ratio_match.group(1))})
            
    if not data_list: return None, None
    df = pd.DataFrame(data_list)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['ratio']].values).astype(str)
    
    np.random.seed(42)
    df['jitter'] = np.random.uniform(-0.5, 0.5, size=len(df))
    return df, kmeans.cluster_centers_

def show_U_page():
    st.title("U 폰트 기둥 두께 비율 클러스터링")
    file_path = "U_result/U_basic_analysis.txt"
    result_folder = "U_result"
    
    df, centroids = load_data_U(file_path)
    if df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(df, x="ratio", y="jitter", color="cluster", hover_data=["filename"],
                             labels={"ratio": "왼쪽 대비 오른쪽 비율 (배)", "cluster": "그룹"})
            fig.add_trace(go.Scatter(x=centroids[:, 0], y=[0]*len(centroids), mode='markers',
                                     marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), name='중심점', hoverinfo='skip'))
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_layout(height=600)
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
            
        with col2:
            st.subheader("선택된 이미지 확인")
            if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
                selected_filename = event.selection.points[0]["customdata"][0]
                orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
                
                st.markdown(f"**원본 파일:** `{orig_fname}`")
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: st.image(orig_img, use_container_width=True)
                else: st.warning("원본 이미지를 찾을 수 없습니다.")
                
                st.divider()
                
                actual_crop = f"{selected_filename}_U_stroke_basic"
                st.markdown(f"**크롭 파일:** `{actual_crop}`")
                crop_img = get_image_from_github(result_folder, actual_crop)
                if crop_img: st.image(crop_img, use_container_width=True)
                else: st.warning("크롭 이미지를 찾을 수 없습니다.")
    else:
        st.error(f"'{file_path}' 데이터를 찾을 수 없거나 내용이 비어있습니다.")

# ==========================================
# 4. 메인 네비게이션
# ==========================================
st.sidebar.title("폰트 분석 메뉴 V2")
menu = st.sidebar.radio("알파벳 선택", ["C 폰트 분석", "G 폰트 분석", "U 폰트 분석"])

if menu == "C 폰트 분석":
    show_CG_page("C")
elif menu == "G 폰트 분석":
    show_CG_page("G")
elif menu == "U 폰트 분석":
    show_U_page()
