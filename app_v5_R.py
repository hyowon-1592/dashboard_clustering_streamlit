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
st.set_page_config(page_title="R 폰트 분석 대시보드", layout="wide")
MAIN_COLOR = '#636EFA' # 클러스터링 제거로 단일 포인트 색상 사용

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
DATA_ROOT = "data_clustering_v5" 

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
        # 파일 타입이 'file'인 항목들의 이름만 추출
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
        
    # 선택된 데이터의 폰트명(앞부분) 추출
    prefixes = sorted(df['filename'].apply(lambda x: x.split(" ")[0]).unique())
    front_file_list = get_front_image_list()
    
    if not front_file_list:
        st.warning("깃허브의 'front_image' 폴더에서 파일 목록을 가져오지 못했습니다.")
        return
        
    # 표 데이터 생성
    table_data = []
    available_files = {} # 드롭다운 선택용 딕셔너리
    
    for prefix in prefixes:
        matched_file = find_front_image(prefix, front_file_list)
        if matched_file:
            table_data.append({"폰트명": prefix, "전면 사진 파일명": matched_file, "상태": "확인 가능"})
            available_files[prefix] = matched_file
        else:
            table_data.append({"폰트명": prefix, "전면 사진 파일명": "-", "상태": "파일 없음"})
            
    res_df = pd.DataFrame(table_data)
    
    # 레이아웃 분할: 왼쪽은 표, 오른쪽은 선택한 이미지 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        
    with col2:
        if available_files:
            # 사용자가 표를 보고 보고싶은 폰트를 선택
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

def show_front_image_grid(df):
    """현재 필터링된 데이터의 전면 사진 모아보기 그리드 렌더링"""
    st.divider()
    st.subheader("현재 화면의 전면 사진 모아보기")
    
    if df.empty:
        st.info("선택된 데이터가 없습니다.")
        return
        
    prefixes = df['filename'].apply(lambda x: x.split(" ")[0]).unique()
    front_file_list = get_front_image_list()
    
    if not front_file_list:
        st.warning("깃허브의 'front_image' 폴더에서 파일 목록을 가져오지 못했습니다.")
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
        st.info("전면 사진 파일이 없습니다.")


# ==========================================
# 2. 데이터 처리 (R 폰트 전용)
# ==========================================
@st.cache_data
def load_data_r_combined(file_path):
    """R의 전체 너비 대비 중앙점 X 거리 비율(%) 추출"""
    text_data = get_text_from_github(file_path)
    if not text_data: return None
    
    data_list = []
    pattern = re.compile(r"\[(.*?)\].*?비율:\s*([\d\.]+)%")
    for line in text_data.split('\n'):
        match = pattern.search(line.strip())
        if match:
            data_list.append({"filename": match.group(1).strip(), "value": float(match.group(2))})
            
    if not data_list: return None
    df = pd.DataFrame(data_list)
    np.random.seed(42)
    df['jitter'] = np.random.uniform(-0.5, 0.5, size=len(df))
    return df

@st.cache_data
def load_data_r_gap(file_path):
    """R의 유격 비율(거리/너비) 추출 및 퍼센트(%)로 변환"""
    text_data = get_text_from_github(file_path)
    if not text_data: return None
    
    data_list = []
    blocks = text_data.split("---------------------------------------------")
    for block in blocks:
        fname_match = re.search(r"파일명:\s*(.+)", block)
        gap_match = re.search(r"▶ 비율\(거리/너비\):\s*([\d\.]+)", block)
        if fname_match and gap_match:
            fname = fname_match.group(1).strip()
            # 추출된 값에 100을 곱하여 퍼센트(%)로 변환
            gap_val = float(gap_match.group(1)) * 100.0
            data_list.append({"filename": fname, "value": gap_val})
            
    if not data_list: return None
    df = pd.DataFrame(data_list)
    np.random.seed(42)
    df['jitter'] = np.random.uniform(-0.5, 0.5, size=len(df))
    return df


# ==========================================
# 3. 페이지 렌더링 함수
# ==========================================
def render_1d_page(title, df, x_label, file_path, result_folder, suffix=""):
    if df is None:
        st.error(f"데이터를 찾을 수 없습니다. 경로('{file_path}')를 확인해주세요.")
        return

    header_col1, header_col2 = st.columns([2, 1])
    header_col1.title(title)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        min_val, max_val = float(df['value'].min()), float(df['value'].max())
        
        # 슬라이더 동적 포맷 설정 (소수점 자릿수 계산)
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
        
        header_col2.markdown(
            f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
            f"전체 데이터: {len(df)}개 | <span style='color: #4CAF50;'>조건 만족: {len(selected_df)}개</span></div>", 
            unsafe_allow_html=True
        )
        
        fig = go.Figure()
        unselected = df[~df['is_selected']]
        
        fig.add_trace(go.Scatter(
            x=unselected['value'], y=unselected['jitter'], mode='markers',
            marker=dict(color='rgba(150, 150, 150, 0.2)', size=6), hoverinfo='skip', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=selected_df['value'], y=selected_df['jitter'], mode='markers',
            marker=dict(color=MAIN_COLOR, size=8),
            name='선택됨', customdata=selected_df[['filename']],
            hovertemplate="<b>%{customdata[0]}</b><br>"+x_label+f": %{{x:.{max_decimals}f}}<extra></extra>"
        ))
        
        fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
                      fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50")
        
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=x_label)
        
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True, key=f"chart_1d_{result_folder}_{suffix}")
            
    with col2:
        st.subheader("선택된 이미지 확인")
        if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
            selected_filename = event.selection.points[0]["customdata"][0]
            orig_fname = get_orig_fname(selected_filename)
            actual_crop = f"{selected_filename}{suffix}"
            
            orig_img = get_image_from_github("Seg_RGB", orig_fname)
            crop_img = get_image_from_github(result_folder, actual_crop)
            
            c1, c2 = st.columns(2)
            if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
            if crop_img: c2.image(crop_img, caption="결과(크롭)", use_container_width=True)
            st.divider()
            
        with st.expander(f"범위 내 데이터 카드 모두 보기 ({len(selected_df)}개)", expanded=False):
            card_container = st.container(height=550)
            for idx, row in selected_df.iterrows():
                font_id = row['filename'].split(' ')[0] 
                orig_fname = get_orig_fname(row['filename'])
                actual_crop = f"{row['filename']}{suffix}"
                
                with card_container.container(border=True):
                    st.markdown(f"<div style='background-color: {MAIN_COLOR}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id}</b></div>", unsafe_allow_html=True)
                    img_c1, img_c2 = st.columns(2)
                    
                    orig_img = get_image_from_github("Seg_RGB", orig_fname)
                    if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                    else: img_c1.caption("원본 없음")
                    
                    crop_img = get_image_from_github(result_folder, actual_crop)
                    if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                    else: img_c2.caption("결과 없음")
                    
                    # 카드 내부 수치도 동적 소수점 적용
                    format_str = f"{x_label}: {{:.{max_decimals}f}}"
                    st.caption(format_str.format(row['value']))
                    
    # show_front_image_grid(selected_df)
    show_front_image_list(selected_df)


st.sidebar.title("R 폰트 분석 필터링")

if "menu" not in st.session_state:
    st.session_state.menu = "R 너비 대비 중심점 비율 분석"

def change_menu(new_menu):
    st.session_state.menu = new_menu

def nav_button(label):
    btn_type = "primary" if st.session_state.menu == label else "secondary"
    st.sidebar.button(label, type=btn_type, use_container_width=True, on_click=change_menu, args=(label,))

st.sidebar.markdown("<br><b>--- [R 폰트 분석] ---</b>", unsafe_allow_html=True)
nav_button("R 너비 대비 중심점 비율 분석")
nav_button("R 유격 비율 분석 (%)")

menu = st.session_state.menu

# ==========================================
# (이하 각 페이지 조건문 로직)
# ==========================================
if menu == "R 너비 대비 중심점 비율 분석":
    folder_name = "R_result_combined"
    file_path = f"{folder_name}/R_analysis_combined.txt"
    df = load_data_r_combined(file_path)
    render_1d_page("R 전체 너비 대비 중심점 X거리 비율(%) 분포", df, "비율 (%)", file_path, folder_name, suffix="_combined")

elif menu == "R 유격 비율 분석 (%)":
    folder_name = "R_result"
    file_path = f"{folder_name}/R_analysis.txt" # os.path.join 대신 / 사용
    df = load_data_r_gap(file_path)
    render_1d_page("R 폰트 유격 비율(%) 분포", df, "유격 비율 (%)", file_path, folder_name, suffix="_R_leg")
