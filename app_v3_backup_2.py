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
st.set_page_config(page_title="통합 폰트 클러스터링 대시보드 v3", layout="wide")

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

# 깃허브 저장소 내의 최상위 데이터 폴더명 (필요 시 수정)
DATA_ROOT = "data_clustering_v3"
COLOR_MAP = {'0': '#EF553B', '1': '#636EFA', '2': '#00CC96', '3': '#AB63FA'}

# ==========================================
# 1. 깃허브 연동 헬퍼 함수
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
# 2. 데이터 처리 (C, G, U, R)
# ==========================================
# [C, G 데이터 로드]
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

# [U 데이터 로드]
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

# [R 데이터 로드 - 두께]
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

# [R 데이터 로드 - 좌표]
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

# [R 데이터 로드 - 너비]
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
# 3. 페이지 렌더링 함수 (C, G, U)
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
            chart_placeholder = st.empty()
            
            slider_col1, slider_col2 = st.columns(2)
            selected_x = slider_col1.slider("X축 (상단 대비 비율) 범위 선택", 0.0, 5.0, (0.0, 5.0), step=0.1)
            selected_y = slider_col2.slider("Y축 (하단 대비 비율) 범위 선택", 0.0, 5.0, (0.0, 5.0), step=0.1)
            
            mask = (df['top_ratio'] >= selected_x[0]) & (df['top_ratio'] <= selected_x[1]) & \
                   (df['bottom_ratio'] >= selected_y[0]) & (df['bottom_ratio'] <= selected_y[1])
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
                x=unselected['top_ratio'], y=unselected['bottom_ratio'], mode='markers',
                marker=dict(color='rgba(150, 150, 150, 0.2)', size=6), hoverinfo='skip', showlegend=False
            ))
            
            for cl in sorted(selected_df['cluster'].unique()):
                c_df = selected_df[selected_df['cluster'] == cl]
                fig.add_trace(go.Scatter(
                    x=c_df['top_ratio'], y=c_df['bottom_ratio'], mode='markers',
                    marker=dict(color=COLOR_MAP.get(cl, '#333'), size=8),
                    name=f'그룹 {cl}', customdata=c_df[['filename']],
                    hovertemplate="<b>%{customdata[0]}</b><br>상단: %{x}<br>하단: %{y}<extra></extra>"
                ))
            
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
                name='중심점', hoverinfo='skip'
            ))
            
            fig.add_shape(
                type="rect", x0=selected_x[0], y0=selected_y[0], x1=selected_x[1], y1=selected_y[1],
                fillcolor="green", opacity=0.15, line=dict(color="#4CAF50", width=2), layer="below"
            )
            
            fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="상단 대비 비율 (배)", yaxis_title="하단 대비 비율 (배)")
            
            with chart_placeholder:
                event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
        with col2:
            st.subheader("선택된 이미지 확인")
            if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
                selected_filename = event.selection.points[0]["customdata"][0]
                orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                actual_crop = f"{selected_filename}_{font_name}_stroke_basic"
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
                actual_crop = f"{row['filename']}_{font_name}_stroke_basic"
                
                with card_container.container(border=True):
                    st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                    img_c1, img_c2 = st.columns(2)
                    
                    orig_img = get_image_from_github("Seg_RGB", orig_fname)
                    if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                    
                    crop_img = get_image_from_github(result_folder, actual_crop)
                    if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                    
                    st.caption(f"상단: {row['top_ratio']:.2f} | 하단: {row['bottom_ratio']:.2f}")
    else:
        st.error("데이터를 찾을 수 없습니다.")

def show_U_page():
    file_path = "U_result/U_basic_analysis.txt"
    result_folder = "U_result"
    
    df, centroids = load_data_U(file_path)
    if df is not None:
        header_col1, header_col2 = st.columns([2, 1])
        header_col1.title("U 폰트 기둥 두께 비율 클러스터링")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_placeholder = st.empty()
            selected_range = st.slider("X축 (비율) 범위 선택", 0.0, 5.0, (0.0, 5.0), step=0.1)
            
            mask = (df['ratio'] >= selected_range[0]) & (df['ratio'] <= selected_range[1])
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
                x=unselected['ratio'], y=unselected['jitter'], mode='markers',
                marker=dict(color='rgba(150, 150, 150, 0.2)', size=6), hoverinfo='skip', showlegend=False
            ))
            
            for cl in sorted(selected_df['cluster'].unique()):
                c_df = selected_df[selected_df['cluster'] == cl]
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
            
            fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
                          fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50")
            
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
            
            with chart_placeholder:
                event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
        with col2:
            st.subheader("선택된 이미지 (그래프 클릭)")
            if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
                selected_filename = event.selection.points[0]["customdata"][0]
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
    else:
        st.error("데이터를 찾을 수 없습니다.")


# ==========================================
# 3. 페이지 렌더링 함수 (R 폰트용)
# ==========================================
def render_1d_page(title, df, centroids, x_label, file_path, result_folder, suffix=""):
    if df is None:
        st.error(f"데이터를 찾을 수 없습니다. 경로('{file_path}')를 확인해주세요.")
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
            actual_crop = f"{selected_filename}{suffix}"
            
            orig_img = get_image_from_github("Seg_RGB", orig_fname)
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
            actual_crop = f"{row['filename']}{suffix}"
            
            with card_container.container(border=True):
                st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                img_c1, img_c2 = st.columns(2)
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                
                crop_img = get_image_from_github(result_folder, actual_crop)
                if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                
                st.caption(f"{x_label}: {row['value']}")

def render_2d_page(title, df, centroids, file_path, result_folder, suffix=""):
    if df is None:
        st.error(f"데이터를 찾을 수 없습니다. 경로('{file_path}')를 확인해주세요.")
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
            type="rect", x0=selected_x[0], y0=selected_y[0], x1=selected_x[1], y1=selected_y[1],
            fillcolor="green", opacity=0.15, line=dict(color="#4CAF50", width=2), layer="below"
        )
        
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="X 좌표", yaxis_title="Y 좌표", yaxis=dict(autorange="reversed"))
        
        with chart_placeholder:
            event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
            
    with col2:
        st.subheader("선택된 이미지 확인")
        if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
            selected_filename = event.selection.points[0]["customdata"][0]
            orig_fname = selected_filename.split('_crop')[0] + '_crop' if '_crop' in selected_filename else selected_filename
            actual_crop = f"{selected_filename}{suffix}"
            
            orig_img = get_image_from_github("Seg_RGB", orig_fname)
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
            actual_crop = f"{row['filename']}{suffix}"
            
            with card_container.container(border=True):
                st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id} (그룹 {row['cluster']})</b></div>", unsafe_allow_html=True)
                img_c1, img_c2 = st.columns(2)
                
                orig_img = get_image_from_github("Seg_RGB", orig_fname)
                if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                
                crop_img = get_image_from_github(result_folder, actual_crop)
                if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                
                st.caption(f"좌표: ({row['x_val']}, {row['y_val']})")










# ==========================================
# 4. 통합 네비게이션 및 메뉴 라우팅
# ==========================================
st.sidebar.title("통합 폰트 분석 대시보드")

# 세션 상태 초기화 (처음 실행 시 기본 화면)
if "menu" not in st.session_state:
    st.session_state.menu = "C 폰트 분석"

# 메뉴 변경 콜백 함수 (버튼 클릭 시 즉각 반영)
def change_menu(new_menu):
    st.session_state.menu = new_menu

# 버튼 렌더링 함수 (선택된 메뉴는 색상 강조)
def nav_button(label):
    btn_type = "primary" if st.session_state.menu == label else "secondary"
    st.sidebar.button(label, type=btn_type, use_container_width=True, on_click=change_menu, args=(label,))

# 구분선은 클릭 불가능한 일반 텍스트(Markdown)로 처리
st.sidebar.markdown("<br><b>--- [C, G, U 분석] ---</b>", unsafe_allow_html=True)
nav_button("C 폰트 분석")
nav_button("G 폰트 분석")
nav_button("U 폰트 분석")
nav_button("C, G, U 공통 그룹 찾기")

st.sidebar.markdown("<br><b>--- [R 폰트 분석] ---</b>", unsafe_allow_html=True)
nav_button("R 다리 stroke 클러스터링")
nav_button("R centre coordinate 클러스터링")
nav_button("R width 클러스터링")
nav_button("R 두께 & 너비 공통 그룹 찾기")

st.sidebar.markdown("<br><b>--- [최종 분석] ---</b>", unsafe_allow_html=True)
nav_button("전체 폰트 교집합 (C, G, U, R)")

# 현재 상태로 menu 변수 덮어쓰기
menu = st.session_state.menu

# ==========================================
# (이하 각 페이지 조건문 로직)
# ==========================================

# [C, G, U 처리]
if menu == "C 폰트 분석":
    show_CG_page("C")
elif menu == "G 폰트 분석":
    show_CG_page("G")
elif menu == "U 폰트 분석":
    show_U_page()
elif menu == "C, G, U 공통 그룹 찾기":
    st.title("C, G, U 폰트 클러스터 교집합 분석")
    st.markdown("C, G, U 폰트 각각의 군집(그룹) 조합을 선택하여 **세 알파벳 모두 지정한 특성 그룹에 속하는 폰트**를 찾습니다.")
    
    df_c, _ = load_data_2D("C_result/C_basic_analysis.txt")
    df_g, _ = load_data_2D("G_result/G_basic_analysis.txt")
    df_u, _ = load_data_U("U_result/U_basic_analysis.txt")
    
    if all(d is not None for d in [df_c, df_g, df_u]):
        for d in [df_c, df_g, df_u]:
            d['orig_fname'] = d['filename'].apply(lambda x: x.split('_crop')[0] + '_crop' if '_crop' in x else x)
            # 🔥 중복 데이터 제거 (C, G, U 병합 시 중복 증식 방지)
            d.drop_duplicates(subset=['orig_fname'], keep='first', inplace=True)
        
        df_c_clean = df_c.rename(columns={'cluster': 'cluster_C', 'top_ratio': 'top_C', 'bottom_ratio': 'bottom_C'})
        df_g_clean = df_g.rename(columns={'cluster': 'cluster_G', 'top_ratio': 'top_G', 'bottom_ratio': 'bottom_G'})
        df_u_clean = df_u.rename(columns={'cluster': 'cluster_U', 'ratio': 'ratio_U'})
        
        merge1 = pd.merge(df_c_clean[['orig_fname', 'cluster_C', 'top_C', 'bottom_C']], 
                          df_g_clean[['orig_fname', 'cluster_G', 'top_G', 'bottom_G']], on='orig_fname')
        merged_df = pd.merge(merge1, df_u_clean[['orig_fname', 'cluster_U', 'ratio_U']], on='orig_fname')
        
        st.subheader("🔍 조합할 그룹 선택")
        col1, col2, col3 = st.columns(3)
        with col1: sel_c = st.selectbox("C 폰트 그룹", ['0', '1', '2'])
        with col2: sel_g = st.selectbox("G 폰트 그룹", ['0', '1', '2'])
        with col3: sel_u = st.selectbox("U 폰트 그룹", ['0', '1', '2'])
        
        common_df = merged_df[(merged_df['cluster_C'] == sel_c) & (merged_df['cluster_G'] == sel_g) & (merged_df['cluster_U'] == sel_u)]
        st.write(f"해당 조합에 공통으로 속하는 폰트는 총 **{len(common_df)}**개 입니다.")
        
        if not common_df.empty:
            st.dataframe(common_df, use_container_width=True)
            with st.expander("선택된 교집합 폰트 원본 이미지 모두 보기", expanded=True):
                cols = st.columns(5)
                col_idx = 0
                for _, row in common_df.iterrows():
                    img = get_image_from_github("Seg_RGB", row['orig_fname'])
                    if img:
                        with cols[col_idx % 5]:
                            st.image(img, caption=row['orig_fname'].split(' ')[0], use_container_width=True)
                        col_idx += 1
                if col_idx == 0: st.write("이미지를 불러올 수 없습니다.")

# [R 파트 처리]
elif menu == "R 다리 stroke 클러스터링":
    folder_name = "R_result_thickness"
    file_path = f"{folder_name}/R_thickness.txt"
    df, centroids = load_data_thickness(file_path)
    render_1d_page("R 폰트 다리 픽셀 두께 분석", df, centroids, "두께 (px)", file_path, folder_name, suffix="_thickness")

elif menu == "R centre coordinate 클러스터링":
    folder_name = "R_result_centre"
    file_path = f"{folder_name}/R_centre.txt"
    df, centroids = load_data_coordinates(file_path)
    render_2d_page("R 폰트 중앙 좌표 분석", df, centroids, file_path, folder_name, suffix="_centre")

elif menu == "R width 클러스터링":
    folder_name = "R_result_width"
    file_path = f"{folder_name}/R_width.txt"
    df, centroids = load_data_width(file_path)
    render_1d_page("R 폰트 너비 분석", df, centroids, "너비 (px)", file_path, folder_name, suffix="_width")

elif menu == "R 두께 & 너비 공통 그룹 찾기":
    st.title("R 폰트 다리 두께 & 너비 교집합 분석")
    st.markdown("K-Means 그룹은 번호가 무작위이므로, 아래 분포표를 확인하고 원하는 조합을 직접 선택해보세요!")
    
    df_thick, _ = load_data_thickness("R_result_thickness/R_thickness.txt")
    df_width, _ = load_data_width("R_result_width/R_width.txt")
    
    if df_thick is not None and df_width is not None:
        # 🔥 중복 데이터 제거 (R 내부 병합 시 중복 증식 방지)
        df_thick.drop_duplicates(subset=['filename'], keep='first', inplace=True)
        df_width.drop_duplicates(subset=['filename'], keep='first', inplace=True)

        merged_df = pd.merge(
            df_thick[['filename', 'cluster', 'value']].rename(columns={'cluster': 'cluster_thick', 'value': 'thick_val'}),
            df_width[['filename', 'cluster', 'value']].rename(columns={'cluster': 'cluster_width', 'value': 'width_val'}),
            on='filename'
        )
        
        st.subheader("전체 교집합 분포 현황")
        cross_tab = pd.crosstab(merged_df['cluster_thick'], merged_df['cluster_width'])
        cross_tab.index.name = "두께 그룹 (행)"
        cross_tab.columns.name = "너비 그룹 (열)"
        st.dataframe(cross_tab, use_container_width=True)
        st.divider()
        
        st.subheader("특정 그룹 조합 이미지 확인")
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1: sel_thick = st.selectbox("다리 두께 그룹 선택", ['0', '1', '2'])
        with sel_col2: sel_width = st.selectbox("너비 그룹 선택", ['0', '1', '2'])
            
        common_df = merged_df[(merged_df['cluster_thick'] == sel_thick) & (merged_df['cluster_width'] == sel_width)]
        st.write(f"**두께 그룹 {sel_thick}** 이면서 **너비 그룹 {sel_width}** 인 폰트는 총 **{len(common_df)}**개 입니다.")
        
        if not common_df.empty:
            st.dataframe(common_df[['filename', 'thick_val', 'width_val']], use_container_width=True)
            with st.expander(f"선택된 교집합 이미지 모두 보기", expanded=True):
                cols = st.columns(5)
                col_idx = 0
                for _, row in common_df.iterrows():
                    orig_fname = row['filename'].split('_crop')[0] + '_crop' if '_crop' in row['filename'] else row['filename']
                    img = get_image_from_github("Seg_RGB", orig_fname)
                    if img:
                        with cols[col_idx % 5]:
                            st.image(img, caption=f"{orig_fname.split(' ')[0]}\n({row['thick_val']}px, {row['width_val']}px)", use_container_width=True)
                        col_idx += 1
    else:
        st.error("데이터를 불러오지 못했습니다. 텍스트 파일 경로를 확인해주세요.")

# [최종 전체 폰트 교집합 처리]
elif menu == "전체 폰트 교집합 (C, G, U, R)":
    st.title("C, G, U, R 폰트 최종 교집합 분석")
    st.markdown("모든 알파벳(C, G, U, R 두께, R 너비)의 특정 군집 조건을 **모두 만족하는 폰트**를 한 번에 필터링합니다.")
    
    # 5가지 데이터 모두 불러오기
    df_c, _ = load_data_2D("C_result/C_basic_analysis.txt")
    df_g, _ = load_data_2D("G_result/G_basic_analysis.txt")
    df_u, _ = load_data_U("U_result/U_basic_analysis.txt")
    df_r_thick, _ = load_data_thickness("R_result_thickness/R_thickness.txt")
    df_r_width, _ = load_data_width("R_result_width/R_width.txt")
    
    if all(d is not None for d in [df_c, df_g, df_u, df_r_thick, df_r_width]):
        
        # 파일명을 통일 (KAI000000 (0)_crop)하여 원활한 병합(Merge) 진행
        for d in [df_c, df_g, df_u, df_r_thick, df_r_width]:
            d['orig_fname'] = d['filename'].apply(lambda x: x.split('_crop')[0] + '_crop' if '_crop' in x else x)
            # 🔥 중복 데이터 제거 (최종 교집합 병합 시 중복 증식 완벽 차단)
            d.drop_duplicates(subset=['orig_fname'], keep='first', inplace=True)
        
        # 이름 간소화 및 병합
        c_sub = df_c[['orig_fname', 'cluster']].rename(columns={'cluster': 'C_group'})
        g_sub = df_g[['orig_fname', 'cluster']].rename(columns={'cluster': 'G_group'})
        u_sub = df_u[['orig_fname', 'cluster']].rename(columns={'cluster': 'U_group'})
        rt_sub = df_r_thick[['orig_fname', 'cluster']].rename(columns={'cluster': 'R_Thick_group'})
        rw_sub = df_r_width[['orig_fname', 'cluster']].rename(columns={'cluster': 'R_Width_group'})
        
        # inner merge를 통해 모든 알파벳의 데이터를 보유한 폰트만 추출
        merged = pd.merge(c_sub, g_sub, on='orig_fname', how='inner')
        merged = pd.merge(merged, u_sub, on='orig_fname', how='inner')
        merged = pd.merge(merged, rt_sub, on='orig_fname', how='inner')
        merged = pd.merge(merged, rw_sub, on='orig_fname', how='inner')
        
        # UI: 5개의 드롭다운 생성
        st.subheader("각 폰트별 조합할 그룹 선택")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: sel_c = st.selectbox("C 폰트 그룹", ['0', '1', '2'])
        with c2: sel_g = st.selectbox("G 폰트 그룹", ['0', '1', '2'])
        with c3: sel_u = st.selectbox("U 폰트 그룹", ['0', '1', '2'])
        with c4: sel_rt = st.selectbox("R 두께 그룹", ['0', '1', '2'])
        with c5: sel_rw = st.selectbox("R 너비 그룹", ['0', '1', '2'])
        
        # 5개 조건으로 최종 필터링
        final_df = merged[(merged['C_group'] == sel_c) & 
                          (merged['G_group'] == sel_g) & 
                          (merged['U_group'] == sel_u) & 
                          (merged['R_Thick_group'] == sel_rt) & 
                          (merged['R_Width_group'] == sel_rw)]
        
        st.success(f"이 5가지 조건을 모두 만족하는 완벽한 교집합 폰트는 총 **{len(final_df)}**개 입니다.")
        
        if not final_df.empty:
            st.dataframe(final_df, use_container_width=True)
            
            with st.expander("최종 교집합 폰트 원본 이미지 모두 보기", expanded=True):
                cols = st.columns(5)
                col_idx = 0
                for _, row in final_df.iterrows():
                    img = get_image_from_github("Seg_RGB", row['orig_fname'])
                    if img:
                        with cols[col_idx % 5]:
                            st.image(img, caption=row['orig_fname'].split(' ')[0], use_container_width=True)
                        col_idx += 1
                if col_idx == 0: st.write("이미지를 불러올 수 없습니다.")
    else:
        st.error("데이터를 전부 불러오지 못했습니다. 각 폴더와 텍스트 파일을 확인해주세요.")
