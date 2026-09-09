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
st.set_page_config(page_title="R 폰트 클러스터링 대시보드 v3", layout="wide")

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

DATA_ROOT = "data_clustering_v3"
COLOR_MAP = {'0': '#EF553B', '1': '#636EFA', '2': '#00CC96', '3': '#AB63FA'}
RESULT_FOLDER = "R_result" # 결과 이미지가 있는 폴더명

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

def show_front_image_grid(df):
    st.divider()
    st.subheader("그룹별 전면 사진 모아보기")
    
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
# 2. 데이터 로더 함수 (정규표현식 파싱)
# ==========================================
@st.cache_data
def load_data_thickness(file_path):
    text_data = get_text_from_github(file_path)
    if not text_data: return None, None
    
    data_list = []
    pattern = re.compile(r"\[(.*?)\]\s*다리 픽셀 두께:\s*([\d\.]+)\s*px")
    for line in text_data.split('\n'):
        match = pattern.search(line.strip())
        if match:
            data_list.append({"filename": match.group(1).strip(), "value": float(match.group(2))})
            
    if not data_list: return None, None
    df = pd.DataFrame(data_list)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['value']].values).astype(str)
    np.random.seed(42)
    df['jitter'] = np.random.uniform(-0.5, 0.5, size=len(df))
    return df, kmeans.cluster_centers_

@st.cache_data
def load_data_coordinates(file_path):
    text_data = get_text_from_github(file_path)
    if not text_data: return None, None
    
    data_list = []
    pattern = re.compile(r"\[(.*?)\]\s*중앙 좌표:\s*\(([\d\.]+),\s*([\d\.]+)\)")
    for line in text_data.split('\n'):
        match = pattern.search(line.strip())
        if match:
            data_list.append({"filename": match.group(1).strip(), "x_val": float(match.group(2)), "y_val": float(match.group(3))})
            
    if not data_list: return None, None
    df = pd.DataFrame(data_list)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['x_val', 'y_val']].values).astype(str)
    return df, kmeans.cluster_centers_

@st.cache_data
def load_data_width(file_path):
    text_data = get_text_from_github(file_path)
    if not text_data: return None, None
    
    data_list = []
    pattern = re.compile(r"\[(.*?)\]\s*너비:\s*([\d\.]+)\s*px")
    for line in text_data.split('\n'):
        match = pattern.search(line.strip())
        if match:
            data_list.append({"filename": match.group(1).strip(), "value": float(match.group(2))})
            
    if not data_list: return None, None
    df = pd.DataFrame(data_list)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['value']].values).astype(str)
    np.random.seed(42)
    df['jitter'] = np.random.uniform(-0.5, 0.5, size=len(df))
    return df, kmeans.cluster_centers_

# ==========================================
# 3. 페이지 렌더링 함수
# ==========================================
def render_1d_page(title, df, centroids, x_label, file_path):
    if df is None:
        st.error(f"데이터를 찾을 수 없습니다. GitHub 경로('{file_path}')를 확인해주세요.")
        return

    header_col1, header_col2 = st.columns([2, 1])
    header_col1.title(title)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        chart_placeholder = st.empty()
        
        min_val, max_val = float(df['value'].min()), float(df['value'].max())
        selected_range = st.slider(f"{x_label} 범위 선택", min_val, max_val, (min_val, max_val))
        
        mask = (df['value'] >= selected_range[0]) & (df['value'] <= selected_range[1])
        df['is_selected'] = mask
        selected_df = df[mask]
        
        header_col2.markdown(
            f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
            f"전체 데이터: {len(df)}개 | <span style='color: #4CAF50;'>선택됨: {len(selected_df)}개</span></div>", 
            unsafe_allow_html=True
        )
        
        fig = go.Figure()
        
        unselected = df[~df['is_selected']]
        fig.add_trace(go.Scatter(
            x=unselected['value'], y=unselected['jitter'], mode='markers',
            marker=dict(color='rgba(150, 150, 150, 0.2)', size=6), hoverinfo='skip', showlegend=False
        ))
        
        for cl in sorted(selected_df['cluster'].unique()):
            c_df = selected_df[selected_df['cluster'] == cl]
            fig.add_trace(go.Scatter(
                x=c_df['value'], y=c_df['jitter'], mode='markers',
                marker=dict(color=COLOR_MAP.get(cl, '#333'), size=8),
                name=f'그룹 {cl}', customdata=c_df[['filename']],
                hovertemplate="<b>%{customdata[0]}</b><br>"+x_label+": %{x}<extra></extra>"
            ))
        
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=[0]*len(centroids), mode='markers',
            marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
            name='중심점', hoverinfo='skip'
        ))
        
        fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
                      fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50")
        
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=x_label)
        
        with chart_placeholder:
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
            
    with col2:
        st.subheader("선택된 이미지 확인")
        
        if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
            selected_filename = event.selection.points[0]["customdata"][0]
            orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
            actual_crop = selected_filename
            
            orig_img = get_image_from_github("Seg_RGB", orig_fname)
            crop_img = get_image_from_github(RESULT_FOLDER, actual_crop)
            
            c1, c2 = st.columns(2)
            if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
            if crop_img: c2.image(crop_img, caption="결과(크롭)", use_container_width=True)
            st.divider()
            
        st.markdown(f"**범위 내 데이터 ({len(selected_df)}개)**")
        card_container = st.container(height=550)
        
        for idx, row in selected_df.iterrows():
            bg_color = COLOR_MAP.get(row['cluster'], '#555')
            font_id = row['filename'].split(' ')[0] 
            orig_fname = row['filename'].split('_crop')[0] + '_crop' if '_crop' in row['filename'] else row['filename']
            actual_crop = row['filename']
            
            with card_container.container(border=True):
                st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                
                img_c1, img_c2 = st.columns(2)
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                
                crop_img = get_image_from_github(RESULT_FOLDER, actual_crop)
                if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                
                st.caption(f"{x_label}: {row['value']}")
            
    show_front_image_grid(df)


def render_2d_page(title, df, centroids, file_path):
    if df is None:
        st.error(f"데이터를 찾을 수 없습니다. GitHub 경로('{file_path}')를 확인해주세요.")
        return

    header_col1, header_col2 = st.columns([2, 1])
    header_col1.title(title)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        chart_placeholder = st.empty()
        
        slider_col1, slider_col2 = st.columns(2)
        min_x, max_x = float(df['x_val'].min()), float(df['x_val'].max())
        min_y, max_y = float(df['y_val'].min()), float(df['y_val'].max())
        
        selected_x = slider_col1.slider("X 좌표 범위", min_x, max_x, (min_x, max_x))
        selected_y = slider_col2.slider("Y 좌표 범위", min_y, max_y, (min_y, max_y))
        
        mask = (df['x_val'] >= selected_x[0]) & (df['x_val'] <= selected_x[1]) & \
               (df['y_val'] >= selected_y[0]) & (df['y_val'] <= selected_y[1])
        df['is_selected'] = mask
        selected_df = df[mask]
        
        header_col2.markdown(
            f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
            f"전체 데이터: {len(df)}개 | <span style='color: #4CAF50;'>선택됨: {len(selected_df)}개</span></div>", 
            unsafe_allow_html=True
        )
        
        fig = go.Figure()
        
        unselected = df[~df['is_selected']]
        fig.add_trace(go.Scatter(
            x=unselected['x_val'], y=unselected['y_val'], mode='markers',
            marker=dict(color='rgba(150, 150, 150, 0.2)', size=6), hoverinfo='skip', showlegend=False
        ))
        
        for cl in sorted(selected_df['cluster'].unique()):
            c_df = selected_df[selected_df['cluster'] == cl]
            fig.add_trace(go.Scatter(
                x=c_df['x_val'], y=c_df['y_val'], mode='markers',
                marker=dict(color=COLOR_MAP.get(cl, '#333'), size=8),
                name=f'그룹 {cl}', customdata=c_df[['filename']],
                hovertemplate="<b>%{customdata[0]}</b><br>X: %{x}<br>Y: %{y}<extra></extra>"
            ))
        
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode='markers',
            marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
            name='중심점', hoverinfo='skip'
        ))
        
        fig.add_shape(
            type="rect",
            x0=selected_x[0], y0=selected_y[0], x1=selected_x[1], y1=selected_y[1],
            fillcolor="green", opacity=0.15, line=dict(color="#4CAF50", width=2), layer="below"
        )
        
        # 화면의 좌표계이므로 y축이 반전(위가 0)일 경우를 대비해 설정 (필요시 autorsize=True 옵션 조정)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="X 좌표", yaxis_title="Y 좌표", yaxis=dict(autorange="reversed"))
        
        with chart_placeholder:
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
            
    with col2:
        st.subheader("선택된 이미지 확인")
        
        if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
            selected_filename = event.selection.points[0]["customdata"][0]
            orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
            actual_crop = selected_filename
            
            orig_img = get_image_from_github("Seg_RGB", orig_fname)
            crop_img = get_image_from_github(RESULT_FOLDER, actual_crop)
            
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
            actual_crop = row['filename']
            
            with card_container.container(border=True):
                st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                
                img_c1, img_c2 = st.columns(2)
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                
                crop_img = get_image_from_github(RESULT_FOLDER, actual_crop)
                if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                
                st.caption(f"좌표: ({row['x_val']}, {row['y_val']})")
            
    show_front_image_grid(df)

# ==========================================
# 4. 메인 네비게이션
# ==========================================
st.sidebar.title("R 폰트 분석 메뉴")
menu = st.sidebar.radio("분석 항목 선택", [
    "다리 stroke 클러스터링", 
    "centre coordinate 클러스터링", 
    "width 클러스터링"
])

# [중요] 실제 GitHub에 존재하는 txt 파일명으로 아래 파일 경로들을 수정해주세요.
FILE_PATH_THICKNESS = f"{RESULT_FOLDER}/R_thickness.txt"  
FILE_PATH_COORD = f"{RESULT_FOLDER}/R_centre.txt"  
FILE_PATH_WIDTH = f"{RESULT_FOLDER}/R_width.txt"

if menu == "다리 픽셀 두께 클러스터링":
    df, centroids = load_data_thickness(FILE_PATH_THICKNESS)
    render_1d_page("R 폰트 다리 픽셀 두께 분석", df, centroids, "두께 (px)", FILE_PATH_THICKNESS)

elif menu == "중앙 좌표 클러스터링":
    df, centroids = load_data_coordinates(FILE_PATH_COORD)
    render_2d_page("R 폰트 중앙 좌표 분석", df, centroids, FILE_PATH_COORD)

elif menu == "너비 클러스터링":
    df, centroids = load_data_width(FILE_PATH_WIDTH)
    render_1d_page("R 폰트 너비 분석", df, centroids, "너비 (px)", FILE_PATH_WIDTH)
