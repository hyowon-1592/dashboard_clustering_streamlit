import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
import io

# ==========================================
# 0. 페이지 및 깃허브 설정
# ==========================================
st.set_page_config(page_title="R 폰트 대시보드", layout="wide")

# Label에 따른 메인 컬러 지정 (파랑, 빨강)
COLOR_LABEL1 = '#636EFA' # 파랑
COLOR_LABEL2 = '#EF553B' # 빨강
COLOR_L1_DIM = 'rgba(99, 110, 250, 0.2)'
COLOR_L2_DIM = 'rgba(239, 85, 59, 0.2)'

# Streamlit Secrets에서 GitHub 정보 가져오기
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_OWNER = st.secrets["REPO_OWNER"] # 깃허브 아이디
    REPO_NAME = st.secrets["REPO_NAME"]   # 레포지토리 이름
    BRANCH = st.secrets.get("BRANCH", "main")
except KeyError:
    st.error("Streamlit Secrets 설정이 누락되었습니다. 깃허브 토큰과 레포지토리 정보를 설정해주세요.")
    st.stop()

# GitHub API 요청 헤더
HEADERS_API = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# GitHub Raw Content 요청 헤더
HEADERS_RAW = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.raw"
}

# 데이터가 들어있는 최상위 폴더 경로 지정 (경로가 없으면 "" 로 설정)
DATA_ROOT = "data_clustering_v56_R_compared" 

def build_repo_path(sub_path):
    """DATA_ROOT와 하위 경로를 안전하게 결합"""
    if DATA_ROOT:
        return f"{DATA_ROOT}/{sub_path}".replace("//", "/")
    return sub_path

# ==========================================
# 1. 깃허브 연동 헬퍼 함수
# ==========================================
def get_orig_fname(filename):
    """중복되는 원본 파일명 추출 로직"""
    return filename.split('_crop')[0] + '_crop' if '_crop' in filename else filename

@st.cache_data(show_spinner=False, max_entries=300)
def get_image_from_github(folder_name, file_name_without_ext):
    """GitHub Private Repo에서 이미지를 다운로드하여 PIL Image로 반환"""
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        file_path = build_repo_path(f"{folder_name}/{file_name_without_ext}{ext}")
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{file_path}"
        
        response = requests.get(url, headers=HEADERS_RAW)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    return None

@st.cache_data(show_spinner=False)
def get_text_from_github(file_path):
    """GitHub Private Repo에서 텍스트 파일을 읽어오기"""
    full_path = build_repo_path(file_path)
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{full_path}"
    
    response = requests.get(url, headers=HEADERS_RAW)
    if response.status_code == 200:
        return response.text
    return None

@st.cache_data(show_spinner=False)
def get_front_image_list():
    """GitHub API를 통해 front_image 폴더의 파일 목록 가져오기"""
    folder_path = build_repo_path("front_image")
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{folder_path}?ref={BRANCH}"
    
    response = requests.get(api_url, headers=HEADERS_API)
    if response.status_code == 200:
        data = response.json()
        return [item['name'] for item in data if item['type'] == 'file']
    return []

def find_front_image(prefix, file_list):
    for file_name in file_list:
        if file_name.startswith(prefix) and (file_name.endswith('.jpg') or file_name.endswith('.JPG')):
            return file_name
    return None

def show_front_image_list(df):
    """현재 필터링된 데이터의 전면 사진 목록 표 및 개별 확인"""
    st.divider()
    st.subheader("현재 화면의 전면 사진 목록")
    
    if df.empty:
        st.info("선택된 데이터가 없습니다.")
        return
        
    prefixes = sorted(df['filename'].apply(lambda x: x.split(" ")[0]).unique())
    front_file_list = get_front_image_list()
    
    if not front_file_list:
        st.warning("깃허브의 'front_image' 폴더에서 파일 목록을 가져오지 못했습니다.")
        return
        
    table_data = []
    available_files = {} 
    
    for prefix in prefixes:
        matched_file = find_front_image(prefix, front_file_list)
        if matched_file:
            table_data.append({"감정번호": prefix, "전면 사진 파일명": matched_file, "상태": "확인 가능"})
            available_files[prefix] = matched_file
        else:
            table_data.append({"감정번호": prefix, "전면 사진 파일명": "-", "상태": "파일 없음"})
            
    res_df = pd.DataFrame(table_data)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        
    with col2:
        if available_files:
            selected_font = st.selectbox("전면 사진 확인하기 (해당 감정번호 선택 시 전면 사진 확인 가능)", list(available_files.keys()))
            if selected_font:
                img_file = available_files[selected_font]
                with st.spinner("이미지 불러오는 중..."):
                    img = get_front_image_from_github(img_file)
                if img:
                    st.image(img, caption=f"[{selected_font}] {img_file}", use_container_width=True)
                else:
                    st.error("이미지를 불러오지 못했습니다.")
        else:
            st.info("현재 목록에 확인할 수 있는 전면 사진이 없습니다.")
            
@st.cache_data(show_spinner=False, max_entries=50)
def get_front_image_from_github(file_name):
    """전면 이미지를 GitHub에서 직접 가져오기"""
    file_path = build_repo_path(f"front_image/{file_name}")
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{file_path}"
    
    response = requests.get(url, headers=HEADERS_RAW)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

# ==========================================
# 2. 데이터 처리 (R 폰트 라벨 비교 전용)
# ==========================================

@st.cache_data(show_spinner=False)
def load_data_r_compared(file_path1, file_path2):
    """Label 1, 2 두 개의 데이터를 읽고 교집합을 구한 뒤 합쳐서 반환"""
    text_data1 = get_text_from_github(file_path1)
    text_data2 = get_text_from_github(file_path2)
    
    if not text_data1 or not text_data2: 
        st.error(f"깃허브에서 텍스트 파일을 읽어오지 못했습니다. 경로를 확인해주세요.")
        return None
    
    def parse_text(text):
        data = []
        pattern = re.compile(r"\[(.*?)\].*?비율:\s*([\d\.]+)%")
        for line in text.split('\n'):
            match = pattern.search(line.strip())
            if match:
                data.append({"filename": match.group(1).strip(), "value": float(match.group(2))})
        return pd.DataFrame(data)

    df1 = parse_text(text_data1)
    df2 = parse_text(text_data2)
    
    if df1.empty or df2.empty: 
        return None

    # 🔥 [핵심 추가] 파일명에서 '_crop' 앞부분(기본 폰트명)만 추출하여 새로운 열(base_name) 생성
    df1['base_name'] = df1['filename'].apply(lambda x: x.split('_crop')[0].strip())
    df2['base_name'] = df2['filename'].apply(lambda x: x.split('_crop')[0].strip())

    # base_name을 기준으로 교집합 찾기
    common_bases = set(df1['base_name']).intersection(set(df2['base_name']))
    
    if len(common_bases) == 0:
        st.error("❌ 파일명의 '_crop' 앞부분을 기준으로 비교했지만 겹치는 파일이 없습니다.")
        return None
    
    # 공통된 base_name을 가진 데이터만 남기기
    df1 = df1[df1['base_name'].isin(common_bases)].copy()
    df2 = df2[df2['base_name'].isin(common_bases)].copy()
    
    df1['label'] = 'Label 1'
    df2['label'] = 'Label 2'
    
    # 두 라벨의 데이터를 하나로 결합
    df = pd.concat([df1, df2], ignore_index=True)
    
    # 그래프 시각화를 위해 jitter 부여
    np.random.seed(42)
    df.loc[df['label'] == 'Label 1', 'jitter'] = np.random.uniform(0.6, 1.4, size=len(df1))
    df.loc[df['label'] == 'Label 2', 'jitter'] = np.random.uniform(-0.4, 0.4, size=len(df2))
    
    return df

# # ==========================================
# # 3. 페이지 렌더링 함수 (비교 버전)
# # ==========================================
# def render_1d_page_compared(title, df, x_label, folder1, folder2, suffix=""):
#     if df is None:
#         return

#     header_col1, header_col2 = st.columns([2, 1])
#     header_col1.title(title)
    
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         min_val, max_val = float(df['value'].min()), float(df['value'].max())
        
#         # 슬라이더 동적 포맷 설정
#         max_decimals = df['value'].astype(str).apply(lambda x: len(x.split('.')[1]) if '.' in x else 0).max()
#         max_decimals = min(max_decimals, 2)
        
#         val_range = max_val - min_val
#         min_step = 1 / (10 ** max_decimals) if max_decimals > 0 else 0.01
#         step_size = val_range / 100.0 if val_range > 0 else min_step
        
#         selected_range = st.slider(
#             f"{x_label} 범위 필터", 
#             min_value=min_val, 
#             max_value=max_val, 
#             value=(min_val, max_val),
#             step=step_size,
#             format=f"%.{max_decimals}f"
#         )
        
#         mask = (df['value'] >= selected_range[0]) & (df['value'] <= selected_range[1])
#         df['is_selected'] = mask
#         selected_df = df[mask]
        
#         # 고유 데이터(폰트) 기준 개수 계산
#         unique_total = len(df['filename'].unique())
#         unique_selected = len(selected_df['filename'].unique())

#         header_col2.markdown(
#             f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
#             f"전체 폰트: {unique_total}개 | <span style='color: #4CAF50;'>조건 만족: {unique_selected}개</span></div>", 
#             unsafe_allow_html=True
#         )
        
#         # Plotly 그래프 생성
#         fig = go.Figure()
        
#         color_map = {'Label 1': COLOR_LABEL1, 'Label 2': COLOR_LABEL2}
#         dim_map = {'Label 1': COLOR_L1_DIM, 'Label 2': COLOR_L2_DIM}
        
#         for lbl in ['Label 1', 'Label 2']:
#             lbl_df = df[df['label'] == lbl]
#             unselected = lbl_df[~lbl_df['is_selected']]
#             selected = lbl_df[lbl_df['is_selected']]
            
#             # 선택되지 않은 점들 (연하게)
#             fig.add_trace(go.Scatter(
#                 x=unselected['value'], y=unselected['jitter'], mode='markers',
#                 marker=dict(color=dim_map[lbl], size=6), hoverinfo='skip', showlegend=False
#             ))
            
#             # 선택된 점들 (진하게)
#             fig.add_trace(go.Scatter(
#                 x=selected['value'], y=selected['jitter'], mode='markers',
#                 marker=dict(color=color_map[lbl], size=8),
#                 name=lbl, customdata=selected[['filename', 'label']],
#                 hovertemplate="<b>%{customdata[0]}</b><br>Label: %{customdata[1]}<br>"+x_label+f": %{{x:.{max_decimals}f}}<extra></extra>"
#             ))
            
#         fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
#                       fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50")
        
#         fig.update_yaxes(visible=False, showticklabels=False)
#         fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=x_label, legend_title="Labels")
        
#         # 차트 렌더링
#         event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", width="stretch", key=f"chart_1d_compared")
            
#     with col2:
#         st.subheader("선택된 이미지 확인")
#         if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
#             selected_filename = event.selection.points[0]["customdata"][0]
#             selected_label = event.selection.points[0]["customdata"][1]
#             orig_fname = get_orig_fname(selected_filename)
#             actual_crop = f"{selected_filename}{suffix}"
            
#             # Label에 맞는 폴더 선택
#             target_folder = folder1 if selected_label == 'Label 1' else folder2
            
#             orig_img = get_image_from_github("Seg_RGB", orig_fname)
#             crop_img = get_image_from_github(target_folder, actual_crop)
            
#             label_color = COLOR_LABEL1 if selected_label == 'Label 1' else COLOR_LABEL2
#             st.markdown(f"**현재 선택 라벨:** <span style='color:{label_color};'>{selected_label}</span>", unsafe_allow_html=True)
            
#             c1, c2 = st.columns(2)
#             if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
#             if crop_img: c2.image(crop_img, caption=f"결과(크롭)", use_container_width=True)
#             st.divider()
            
#         with st.expander(f"범위 내 데이터 카드 모두 보기 (총 {len(selected_df)}건)", expanded=False):
#             card_container = st.container(height=550)
            
#             # 정렬하여 같은 폰트의 Label 1, Label 2가 인접하게 표시되도록 설정
#             sorted_selected_df = selected_df.sort_values(by=['filename', 'label'])
            
#             for idx, row in sorted_selected_df.iterrows():
#                 font_id = row['filename'].split(' ')[0] 
#                 orig_fname = get_orig_fname(row['filename'])
#                 actual_crop = f"{row['filename']}{suffix}"
                
#                 # 라벨별 카드 색상 및 폴더 분기
#                 card_color = COLOR_LABEL1 if row['label'] == 'Label 1' else COLOR_LABEL2
#                 target_folder = folder1 if row['label'] == 'Label 1' else folder2
                
#                 with card_container.container(border=True):
#                     st.markdown(f"<div style='background-color: {card_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>[{row['label']}] {font_id}</b></div>", unsafe_allow_html=True)
#                     img_c1, img_c2 = st.columns(2)
                    
#                     orig_img = get_image_from_github("Seg_RGB", orig_fname)
#                     if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
#                     else: img_c1.caption("원본 없음")
                    
#                     crop_img = get_image_from_github(target_folder, actual_crop)
#                     if crop_img: img_c2.image(crop_img, caption=f"{row['label']} 결과", use_container_width=True)
#                     else: img_c2.caption("결과 없음")
                    
#                     format_str = f"{x_label}: {{:.{max_decimals}f}}"
#                     st.caption(format_str.format(row['value']))
                    
#     show_front_image_list(selected_df)




def show_comparison_table(df, max_decimals):
    """선택된 데이터의 Label 1과 Label 2 수치를 비교하는 표 렌더링"""
    st.divider()
    st.subheader("===라벨 수치 비교표===")
    
    if df.empty:
        st.info("선택된 데이터가 없습니다.")
        return
        
    # base_name(파일명) 기준으로 Label 1과 Label 2 데이터를 양옆으로 펼치기 (Pivot)
    pivot_df = df.pivot_table(index='base_name', columns='label', values='value', aggfunc='first').reset_index()
    pivot_df.columns.name = None # 인덱스 이름 정리
    
    # 열 이름 변경
    rename_dict = {'base_name': '파일명'}
    if 'Label 1' in pivot_df.columns:
        rename_dict['Label 1'] = 'Label 1 수치'
    if 'Label 2' in pivot_df.columns:
        rename_dict['Label 2'] = 'Label 2 수치'
        
    pivot_df = pivot_df.rename(columns=rename_dict)
    
    # 범위 필터링 때문에 한쪽 라벨만 선택된 경우, 빈칸(NaN)을 '-'로 처리
    pivot_df = pivot_df.fillna("-")
    
    # 소수점 자릿수 깔끔하게 맞추기
    for col in ['Label 1 수치', 'Label 2 수치']:
        if col in pivot_df.columns:
            pivot_df[col] = pivot_df[col].apply(
                lambda x: f"{x:.{max_decimals}f}" if isinstance(x, (int, float)) else x
            )
            
    # 화면에 표 출력
    st.dataframe(pivot_df, use_container_width=True, hide_index=True)


# ==========================================
# 3. 페이지 렌더링 함수 (비교 버전)
# ==========================================
def render_1d_page_compared(title, df, x_label, folder1, folder2, suffix=""):
    if df is None:
        return

    header_col1, header_col2 = st.columns([2, 1])
    header_col1.title(title)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        min_val, max_val = float(df['value'].min()), float(df['value'].max())
        
        # 슬라이더 동적 포맷 설정
        max_decimals = df['value'].astype(str).apply(lambda x: len(x.split('.')[1]) if '.' in x else 0).max()
        max_decimals = min(max_decimals, 2)
        
        val_range = max_val - min_val
        min_step = 1 / (10 ** max_decimals) if max_decimals > 0 else 0.01
        step_size = val_range / 100.0 if val_range > 0 else min_step
        
        selected_range = st.slider(
            f"{x_label} 범위 필터", 
            min_value=min_val, 
            max_value=max_val, 
            value=(min_val, max_val),
            step=step_size,
            format=f"%.{max_decimals}f"
        )
        
        mask = (df['value'] >= selected_range[0]) & (df['value'] <= selected_range[1])
        df['is_selected'] = mask
        selected_df = df[mask]
        
        # 고유 데이터(폰트) 기준 개수 계산
        unique_total = len(df['base_name'].unique())
        unique_selected = len(selected_df['base_name'].unique())

        header_col2.markdown(
            f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
            f"전체 폰트: {unique_total}개 | <span style='color: #4CAF50;'>조건 만족: {unique_selected}개</span></div>", 
            unsafe_allow_html=True
        )
        
        # Plotly 그래프 생성
        fig = go.Figure()
        
        color_map = {'Label 1': COLOR_LABEL1, 'Label 2': COLOR_LABEL2}
        dim_map = {'Label 1': COLOR_L1_DIM, 'Label 2': COLOR_L2_DIM}
        
        for lbl in ['Label 1', 'Label 2']:
            lbl_df = df[df['label'] == lbl]
            unselected = lbl_df[~lbl_df['is_selected']]
            selected = lbl_df[lbl_df['is_selected']]
            
            # 선택되지 않은 점들 (연하게)
            fig.add_trace(go.Scatter(
                x=unselected['value'], y=unselected['jitter'], mode='markers',
                marker=dict(color=dim_map[lbl], size=6), hoverinfo='skip', showlegend=False
            ))
            
            # 선택된 점들 (진하게)
            fig.add_trace(go.Scatter(
                x=selected['value'], y=selected['jitter'], mode='markers',
                marker=dict(color=color_map[lbl], size=8),
                name=lbl, customdata=selected[['filename', 'label']],
                hovertemplate="<b>%{customdata[0]}</b><br>Label: %{customdata[1]}<br>"+x_label+f": %{{x:.{max_decimals}f}}<extra></extra>"
            ))
            
        fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
                      fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50")
        
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=x_label, legend_title="Labels")
        
        # 차트 렌더링
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", width="stretch", key=f"chart_1d_compared")
            
    with col2:
        st.subheader("선택된 이미지 확인")
        if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
            selected_filename = event.selection.points[0]["customdata"][0]
            selected_label = event.selection.points[0]["customdata"][1]
            orig_fname = get_orig_fname(selected_filename)
            actual_crop = f"{selected_filename}{suffix}"
            
            # Label에 맞는 폴더 선택
            target_folder = folder1 if selected_label == 'Label 1' else folder2
            
            orig_img = get_image_from_github("Seg_RGB", orig_fname)
            crop_img = get_image_from_github(target_folder, actual_crop)
            
            label_color = COLOR_LABEL1 if selected_label == 'Label 1' else COLOR_LABEL2
            st.markdown(f"**현재 선택 라벨:** <span style='color:{label_color};'>{selected_label}</span>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
            if crop_img: c2.image(crop_img, caption=f"결과(크롭)", use_container_width=True)
            st.divider()
            
        with st.expander(f"범위 내 데이터 카드 모두 보기 (총 {len(selected_df)}건)", expanded=False):
            card_container = st.container(height=550)
            
            # 정렬하여 같은 폰트의 Label 1, Label 2가 인접하게 표시되도록 설정
            sorted_selected_df = selected_df.sort_values(by=['base_name', 'label'])
            
            for idx, row in sorted_selected_df.iterrows():
                font_id = row['base_name'] 
                orig_fname = get_orig_fname(row['filename'])
                actual_crop = f"{row['filename']}{suffix}"
                
                # 라벨별 카드 색상 및 폴더 분기
                card_color = COLOR_LABEL1 if row['label'] == 'Label 1' else COLOR_LABEL2
                target_folder = folder1 if row['label'] == 'Label 1' else folder2
                
                with card_container.container(border=True):
                    st.markdown(f"<div style='background-color: {card_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>[{row['label']}] {font_id}</b></div>", unsafe_allow_html=True)
                    img_c1, img_c2 = st.columns(2)
                    
                    orig_img = get_image_from_github("Seg_RGB", orig_fname)
                    if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                    else: img_c1.caption("원본 없음")
                    
                    crop_img = get_image_from_github(target_folder, actual_crop)
                    if crop_img: img_c2.image(crop_img, caption=f"{row['label']} 결과", use_container_width=True)
                    else: img_c2.caption("결과 없음")
                    
                    format_str = f"{x_label}: {{:.{max_decimals}f}}"
                    st.caption(format_str.format(row['value']))
                    
    # 전면 사진 대신 비교 표 렌더링 함수 호출
    show_comparison_table(selected_df, max_decimals)

st.sidebar.title("R 폰트(compared)")

if "menu" not in st.session_state:
    st.session_state.menu = "R 너비 대비 중심점 비율 분석 (%)"

def change_menu(new_menu):
    st.session_state.menu = new_menu

def nav_button(label):
    btn_type = "primary" if st.session_state.menu == label else "secondary"
    st.sidebar.button(label, type=btn_type, width="stretch", on_click=change_menu, args=(label,))

st.sidebar.markdown("<br><b>--- [R 폰트 분석] ---</b>", unsafe_allow_html=True)
nav_button("R 너비 대비 중심점 비율 분석 (%)")

menu = st.session_state.menu

# ==========================================
# 실행부 (폴더명 변경 필요)
# ==========================================
if menu == "R 너비 대비 중심점 비율 분석 (%)":
    # 🔥 실제 깃허브 구조에 맞게 label 1과 label 2 폴더명을 수정해주세요.
    folder1 = "R_result_combined_1" 
    folder2 = "R_result_combined_2" 
    
    file_path1 = f"{folder1}/R_analysis_combined.txt"
    file_path2 = f"{folder2}/R_analysis_combined.txt"
    
    df = load_data_r_compared(file_path1, file_path2)
    render_1d_page_compared("R 너비 대비 중심점 X거리 비율(%) 비교 분포(label 1: 강남팀 누끼, label 2: 훈님 누끼)", df, "비율 (%)", folder1, folder2, suffix="_combined")
