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

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = st.secrets["REPO_OWNER"]
    REPO_NAME = st.secrets["REPO_NAME"]
    BRANCH = st.secrets.get("BRANCH", "main")
except KeyError:
    st.error("Streamlit Secrets 설정이 누락되었습니다. 깃허브 토큰과 레포지토리 정보를 설정해주세요.")
    st.stop()

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}
RAW_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.raw"
}

DATA_ROOT = "data_clustering_v2"
COLOR_MAP = {'0': '#EF553B', '1': '#636EFA', '2': '#00CC96', '3': '#AB63FA'}

# ==========================================
# 1. 공통 헬퍼 함수 (GitHub 연동)
# ==========================================
@st.cache_data(show_spinner=False)
def get_image_from_github(folder_name, file_name_without_ext):
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        file_path = f"{DATA_ROOT}/{folder_name}/{file_name_without_ext}{ext}"
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{file_path}"
        response = requests.get(url, headers=RAW_HEADERS)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    return None

@st.cache_data(show_spinner=False)
def get_text_from_github(file_path):
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{DATA_ROOT}/{file_path}"
    response = requests.get(url, headers=RAW_HEADERS)
    if response.status_code == 200:
        return response.text
    return None

@st.cache_data(show_spinner=False)
def get_front_image_list():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{DATA_ROOT}/front_image?ref={BRANCH}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return [file['name'] for file in response.json() if file['type'] == 'file']
    return []

def find_front_image(prefix, file_list):
    for file_name in file_list:
        if file_name.startswith(prefix) and (file_name.endswith('.jpg') or file_name.endswith('.JPG')):
            return file_name
    return None

@st.cache_data(show_spinner=False)
def get_front_image_from_github(file_name):
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{DATA_ROOT}/front_image/{file_name}"
    response = requests.get(url, headers=RAW_HEADERS)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

# ==========================================
# 2. 데이터 처리 및 공통 UI 컴포넌트
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

def show_front_image_grid(df):
    st.divider()
    st.subheader("📦 그룹별 전면 사진 모아보기")
    
    groups = sorted(df['cluster'].unique())
    selected_group = st.radio("확인할 그룹을 선택하세요:", groups, horizontal=True)
    
    if selected_group:
        prefixes = df[df['cluster'] == selected_group]['filename'].apply(lambda x: x.split(" ")[0]).unique()
        front_file_list = get_front_image_list()
        
        if not front_file_list:
            st.warning("GitHub 저장소의 'front_image' 폴더에서 파일 목록을 가져오지 못했습니다.")
            return
            
        cols = st.columns(5)
        col_idx = 0
        
        for prefix in prefixes:
            matched_file = find_front_image(prefix, front_file_list)
            if matched_file:
                img = get_front_image_from_github(matched_file)
                if img:
                    with cols[col_idx % 5]:
                        st.image(img, use_container_width=True, caption=matched_file)
                    col_idx += 1
        
        if col_idx == 0:
            st.info("해당 그룹의 전면 사진 파일이 없습니다.")

# ==========================================
# 3. 폰트별 페이지 렌더링
# ==========================================
def show_CG_page(font_name):
    result_folder = f"{font_name}_result"
    file_path = f"{result_folder}/{font_name}_basic_analysis.txt"
    
    df, centroids = load_data_2D(file_path)
    if df is not None:
        header_col1, header_col2 = st.columns([2, 1])
        header_col1.title(f"{font_name} 폰트 상단/하단 두께 비율 클러스터링")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 **마우스로 그래프 위를 드래그**하여 원하는 영역의 데이터를 선택해보세요!")
            fig = go.Figure()
            
            # 클러스터별 데이터 렌더링
            for cl in sorted(df['cluster'].unique()):
                c_df = df[df['cluster'] == cl]
                fig.add_trace(go.Scatter(
                    x=c_df['top_ratio'], y=c_df['bottom_ratio'], mode='markers',
                    marker=dict(color=COLOR_MAP.get(cl, '#333'), size=8),
                    name=f'그룹 {cl}', customdata=c_df[['filename']],
                    hovertemplate="<b>%{customdata[0]}</b><br>상단: %{x}<br>하단: %{y}<extra></extra>"
                ))
            
            # 중심점 렌더링
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
                name='중심점', hoverinfo='skip'
            ))
            
            # 마우스 드래그를 기본 모드로 설정 (dragmode='select')
            fig.update_layout(
                height=600, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="상단 대비 비율 (배)", yaxis_title="하단 대비 비율 (배)",
                dragmode='select' 
            )
            
            # on_select 속성으로 드래그한 점들의 데이터를 받아옴
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
        with col2:
            # 드래그로 선택된 데이터 추출
            selected_filenames = []
            if event and len(event.selection.points) > 0:
                selected_filenames = [pt["customdata"][0] for pt in event.selection.points if "customdata" in pt]
            
            # 선택된 데이터가 있으면 필터링, 없으면 전체 표시
            selected_df = df[df['filename'].isin(selected_filenames)] if selected_filenames else df
            
            header_col2.markdown(
                f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
                f"전체 데이터: {len(df)}개 | <span style='color: #4CAF50;'>선택됨: {len(selected_df)}개</span></div>", 
                unsafe_allow_html=True
            )
            
            # 점 1개를 '클릭'했을 때 상단에 상세 이미지 프리뷰 띄우기
            if len(selected_filenames) == 1:
                st.subheader("선택된 이미지 상세 보기")
                selected_filename = selected_filenames[0]
                orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                actual_crop = f"{selected_filename}_{font_name}_stroke_basic"
                crop_img = get_image_from_github(result_folder, actual_crop)
                
                c1, c2 = st.columns(2)
                if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
                if crop_img: c2.image(crop_img, caption="결과(크롭)", use_container_width=True)
                st.divider()
                
            # 데이터 카드 그리드 (드래그한 데이터 모두 표시)
            st.markdown(f"**범위 내 데이터 카드 ({len(selected_df)}개)**")
            card_container = st.container(height=550)
            
            for idx, row in selected_df.iterrows():
                bg_color = COLOR_MAP.get(row['cluster'], '#555')
                font_id = row['filename'].split(' ')[0]
                orig_fname = row['filename'].split('_crop')[0] + '_crop' if '_crop' in row['filename'] else row['filename']
                actual_crop = f"{row['filename']}_{font_name}_stroke_basic"
                
                with card_container.container(border=True):
                    st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                    
                    img_c1, img_c2 = st.columns(2)
                    
                    orig_img = get_image_from_github("Seg_RGB", orig_fname)
                    if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                    else: img_c1.caption("원본 없음")
                    
                    crop_img = get_image_from_github(result_folder, actual_crop)
                    if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                    else: img_c2.caption("결과 없음")
                    
                    st.caption(f"상단: {row['top_ratio']:.2f} | 하단: {row['bottom_ratio']:.2f}")
                
        show_front_image_grid(df)
    else:
        st.error(f"데이터를 찾을 수 없습니다.")

def show_U_page():
    file_path = "U_result/U_basic_analysis.txt"
    result_folder = "U_result"
    
    df, centroids = load_data_U(file_path)
    if df is not None:
        header_col1, header_col2 = st.columns([2, 1])
        header_col1.title("U 폰트 기둥 두께 비율 클러스터링")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 **마우스로 그래프 위를 드래그**하여 원하는 영역의 데이터를 선택해보세요!")
            fig = go.Figure()
            
            for cl in sorted(df['cluster'].unique()):
                c_df = df[df['cluster'] == cl]
                fig.add_trace(go.Scatter(
                    x=c_df['ratio'], y=c_df['jitter'], mode='markers',
                    marker=dict(color=COLOR_MAP.get(cl, '#333'), size=8),
                    name=f'그룹 {cl}', customdata=c_df[['filename']],
                    hovertemplate="<b>%{customdata[0]}</b><br>Ratio: %{x}<br>Jitter: %{y}<extra></extra>"
                ))
            
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=[0]*len(centroids), mode='markers',
                marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
                name='중심점', hoverinfo='skip'
            ))
            
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_layout(
                height=600, margin=dict(l=10, r=10, t=30, b=10),
                dragmode='select' # 마우스 드래그를 기본 모드로 설정
            )
            
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
        with col2:
            selected_filenames = []
            if event and len(event.selection.points) > 0:
                selected_filenames = [pt["customdata"][0] for pt in event.selection.points if "customdata" in pt]
            
            selected_df = df[df['filename'].isin(selected_filenames)] if selected_filenames else df
            
            header_col2.markdown(
                f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
                f"전체 데이터: {len(df)}개 | <span style='color: #4CAF50;'>선택됨: {len(selected_df)}개</span></div>", 
                unsafe_allow_html=True
            )
            
            if len(selected_filenames) == 1:
                st.subheader("선택된 이미지 상세 보기")
                selected_filename = selected_filenames[0]
                orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                actual_crop = f"{selected_filename}_U_stroke_basic"
                crop_img = get_image_from_github(result_folder, actual_crop)
                
                c1, c2 = st.columns(2)
                if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
                if crop_img: c2.image(crop_img, caption="결과(크롭)", use_container_width=True)
                st.divider()
                
            st.markdown(f"**범위 내 데이터 카드 ({len(selected_df)}개)**")
            card_container = st.container(height=550)
            
            for idx, row in selected_df.iterrows():
                bg_color = COLOR_MAP.get(row['cluster'], '#555')
                font_id = row['filename'].split(' ')[0] 
                orig_fname = row['filename'].split('_crop')[0] + '_crop' if '_crop' in row['filename'] else row['filename']
                actual_crop = f"{row['filename']}_U_stroke_basic"
                
                with card_container.container(border=True):
                    st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                    
                    img_c1, img_c2 = st.columns(2)
                    
                    orig_img = get_image_from_github("Seg_RGB", orig_fname)
                    if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                    else: img_c1.caption("원본 없음")
                    
                    crop_img = get_image_from_github(result_folder, actual_crop)
                    if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                    else: img_c2.caption("결과 없음")
                    
                    st.caption(f"X (비율): {row['ratio']:.2f} | Y (분산): {row['jitter']:.2f}")
                
        show_front_image_grid(df)
    else:
        st.error(f"데이터를 찾을 수 없습니다.")

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
