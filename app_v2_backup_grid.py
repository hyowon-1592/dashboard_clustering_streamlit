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
    st.title(f"{font_name} 폰트 상단/하단 두께 비율 클러스터링")
    
    result_folder = f"{font_name}_result"
    file_path = f"{result_folder}/{font_name}_basic_analysis.txt"
    
    df, centroids = load_data_2D(file_path)
    if df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(df, x="top_ratio", y="bottom_ratio", color="cluster", hover_data=["filename"])
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
                
                st.divider()
                actual_crop = f"{selected_filename}_{font_name}_stroke_basic"
                st.markdown(f"**크롭 파일:** `{actual_crop}`")
                crop_img = get_image_from_github(result_folder, actual_crop)
                if crop_img: st.image(crop_img, use_container_width=True)
                
        show_front_image_grid(df)
    else:
        st.error(f"데이터를 찾을 수 없습니다.")

def show_U_page():
    file_path = "U_result/U_basic_analysis.txt"
    result_folder = "U_result"
    
    df, centroids = load_data_U(file_path)
    if df is not None:
        # 클러스터 색상 매핑 (영상과 유사한 톤)
        color_map = {'0': '#EF553B', '1': '#636EFA', '2': '#00CC96', '3': '#AB63FA'}
        
        # 헤더 레이아웃 (제목 + 데이터 개수 통계)
        header_col1, header_col2 = st.columns([2, 1])
        header_col1.title("U 폰트 기둥 두께 비율 클러스터링")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 1. 차트와 슬라이더를 배치 (슬라이더가 차트 아래에 오도록 레이아웃 트릭 사용)
            chart_placeholder = st.empty()
            
            min_val, max_val = 0.0, 5.0
            selected_range = st.slider("X축 (비율) 범위 선택", min_val, max_val, (0.0, 5.0), step=0.1)
            
            # 슬라이더 범위에 따른 데이터 필터링
            mask = (df['ratio'] >= selected_range[0]) & (df['ratio'] <= selected_range[1])
            df['is_selected'] = mask
            selected_df = df[mask]
            
            # 우측 상단에 전체/선택 개수 업데이트
            header_col2.markdown(
                f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
                f"전체 데이터: {len(df)}개 | <span style='color: #4CAF50;'>선택됨: {len(selected_df)}개</span></div>", 
                unsafe_allow_html=True
            )
            
            # 2. 커스텀 Plotly 차트 생성
            fig = go.Figure()
            
            # 선택되지 않은 데이터 (회색 처리)
            unselected = df[~df['is_selected']]
            fig.add_trace(go.Scatter(
                x=unselected['ratio'], y=unselected['jitter'],
                mode='markers',
                marker=dict(color='rgba(150, 150, 150, 0.2)', size=6),
                hoverinfo='skip', showlegend=False
            ))
            
            # 선택된 데이터 (클러스터 색상 처리)
            for cl in sorted(selected_df['cluster'].unique()):
                c_df = selected_df[selected_df['cluster'] == cl]
                fig.add_trace(go.Scatter(
                    x=c_df['ratio'], y=c_df['jitter'], mode='markers',
                    marker=dict(color=color_map.get(cl, '#333'), size=8),
                    name=f'그룹 {cl}', customdata=c_df[['filename']],
                    hovertemplate="<b>%{customdata[0]}</b><br>Ratio: %{x}<br>Jitter: %{y}<extra></extra>"
                ))
            
            # 중심점
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=[0]*len(centroids), mode='markers',
                marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
                name='중심점', hoverinfo='skip'
            ))
            
            # 선택 범위 연두색 박스 하이라이트
            fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
                          fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50")
            
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
            
            # 빈 공간(placeholder)에 차트 렌더링
            with chart_placeholder:
                event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
        with col2:
            st.subheader("선택된 이미지 확인")
            
            # 특정 점을 클릭했을 경우 원본/크롭 이미지를 보여주는 기존 기능 유지
            if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
                selected_filename = event.selection.points[0]["customdata"][0]
                orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: st.image(orig_img, caption=orig_fname, use_container_width=True)
                
                actual_crop = f"{selected_filename}_U_stroke_basic"
                crop_img = get_image_from_github(result_folder, actual_crop)
                if crop_img: st.image(crop_img, caption=actual_crop, use_container_width=True)
                st.divider()
                
            # 영상에 나왔던 '카드 형식 리스트' 렌더링 (스크롤 박스 형태)
            st.markdown(f"**범위 내 데이터 카드 ({len(selected_df)}개)**")
            card_container = st.container(height=550) # 데이터가 많을 시 스크롤 되도록 설정
            card_cols = card_container.columns(2)
            
            for idx, row in selected_df.iterrows():
                col = card_cols[idx % 2]
                bg_color = color_map.get(row['cluster'], '#555')
                # 띄어쓰기 이전 문자(예: KAI0EP599F)만 추출하여 폰트 ID로 사용
                font_id = row['filename'].split(' ')[0] 
                
                # HTML/CSS를 활용한 카드 UI
                card_html = f"""
                <div style="background-color: {bg_color}; opacity: 0.85; border-radius: 8px; padding: 12px; margin-bottom: 10px; text-align: center; color: white; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                    <h5 style="margin:0; color: white;">{font_id}</h5>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #eee; font-weight: 500;">
                        X: {row['ratio']:.2f} &nbsp;|&nbsp; Y: {row['jitter']:.2f}
                    </p>
                </div>
                """
                col.markdown(card_html, unsafe_allow_html=True)
                
        # 전면 사진 그리드 함수 호출 (이전 단계 기능)
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
