import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from PIL import Image
import requests
import io

# ==========================================
# 0. 페이지 및 깃허브 설정
# ==========================================
st.set_page_config(page_title="M0~M5 대시보드", layout="wide")
COLOR_MAP = {'0': '#EF553B', '1': '#636EFA', '2': '#00CC96', '3': '#AB63FA', '4': '#FFA15A'}
DEFAULT_COLOR = '#636EFA' # 클러스터링 안 할 때 쓸 기본 색상

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

# 데이터가 들어있는 최상위 폴더 경로 지정
DATA_ROOT = "data_clustering_v1"

# ==========================================
# 1. 깃허브 연동 및 데이터 파싱 헬퍼 함수
# ==========================================
def get_orig_fname(filename):
    return filename.split('_crop')[0] + '_crop' if '_crop' in filename else filename

@st.cache_data(show_spinner=False, max_entries=300)
def get_image_from_github(folder_name, file_name_without_ext):
    """GitHub Private Repo에서 이미지를 다운로드하여 PIL Image로 반환"""
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        file_path = f"{DATA_ROOT}/{folder_name}/{file_name_without_ext}{ext}"
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{file_path}"
        
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    return None

@st.cache_data(show_spinner=False)
def get_text_from_github(file_path):
    """GitHub Private Repo에서 텍스트 파일을 읽어오기"""
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{DATA_ROOT}/{file_path}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"요청 URL: {url}")
        st.error(f"실패 상태 코드: {response.status_code}")
        st.error(f"깃허브 응답 내용: {response.text}")
        return None

@st.cache_data
def load_and_parse_m_metrics(file_path):
    text = get_text_from_github(file_path)
    if not text: return None
    
    data_dict = {}
    current_metric = None
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('[M') and line.endswith('Result]'):
            current_metric = line[1:3]
            continue
        
        if current_metric and line and not line.startswith('sample') and not line.startswith('===') and not line.startswith('*'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    val = float(parts[-1])
                    sample_name = " ".join(parts[:-1]).strip()
                    if sample_name not in data_dict:
                        data_dict[sample_name] = {}
                    data_dict[sample_name][current_metric] = val
                except ValueError:
                    pass
                    
    if not data_dict: return None
    
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df.rename(columns={'index': 'filename'}, inplace=True)
    return df

# ==========================================
# 2. 개별 지표 렌더링 함수
# ==========================================
def render_metric_page(metric_name, metric_desc, df):
    if metric_name not in df.columns:
        st.error(f"{metric_name} 데이터를 찾을 수 없습니다.")
        return
        
    # 결측치(측정 실패) 데이터 따로 빼두기
    missing_df = df[df[metric_name].isna()][['filename']].copy()
    
    # 정상 측정된 데이터만 필터링
    sub_df = df[['filename', metric_name]].dropna().copy()
    sub_df.rename(columns={metric_name: 'value'}, inplace=True)
    
    if len(sub_df) < 2:
        st.warning("데이터가 부족합니다.")
        return

    header_col1, header_col2 = st.columns([2, 1])
    header_col1.title(f"{metric_name} 지표 분석")
    st.caption(f"**설명:** {metric_desc}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # 셀렉트박스로 클러스터링 개수 선택
        cluster_options = [0, 2, 3, 4, 5]
        selected_k = st.selectbox("분류할 그룹 개수 선택 (0 선택 시 일반 scatter)", cluster_options, index=0)
        
        if selected_k > 0:
            actual_k = min(selected_k, len(sub_df))
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            sub_df['cluster'] = kmeans.fit_predict(sub_df[['value']].values).astype(str)
            centroids = kmeans.cluster_centers_
            
            # 보기 좋게 중심점 크기순 정렬
            order = np.argsort(centroids.flatten())
            mapping = {str(old): str(new) for new, old in enumerate(order)}
            sub_df['cluster'] = sub_df['cluster'].map(mapping)
        else:
            sub_df['cluster'] = '0' # 전체를 단일 그룹으로 취급
            centroids = None
        
        np.random.seed(42)
        sub_df['jitter'] = np.random.uniform(-0.5, 0.5, size=len(sub_df))

        # 범위 선택 슬라이더
        min_val, max_val = float(sub_df['value'].min()), float(sub_df['value'].max())
        selected_range = st.slider(f"{metric_name} 수치 범위 선택", min_val, max_val, (min_val, max_val))
        
        mask = (sub_df['value'] >= selected_range[0]) & (sub_df['value'] <= selected_range[1])
        sub_df['is_selected'] = mask
        selected_df = sub_df[mask]
        
        header_col2.markdown(
            f"<div style='text-align: right; margin-top: 25px; color: #a1c8ff; font-weight: bold;'>"
            f"전체 정상 데이터: {len(sub_df)}개 | <span style='color: #4CAF50;'>선택됨: {len(selected_df)}개</span></div>", 
            unsafe_allow_html=True
        )
        
        fig = go.Figure()
        unselected = sub_df[~sub_df['is_selected']]
        
        # 선택 안 된 데이터 (희미하게)
        fig.add_trace(go.Scatter(
            x=unselected['value'], y=unselected['jitter'], mode='markers',
            marker=dict(color='rgba(150, 150, 150, 0.2)', size=6), hoverinfo='skip', showlegend=False
        ))
        
        # 선택된 데이터 렌더링
        for cl in sorted(selected_df['cluster'].unique()):
            c_df = selected_df[selected_df['cluster'] == cl]
            color = DEFAULT_COLOR if selected_k == 0 else COLOR_MAP.get(cl, '#333')
            group_name = "전체 데이터" if selected_k == 0 else f"그룹 {cl}"
            
            fig.add_trace(go.Scatter(
                x=c_df['value'], y=c_df['jitter'], mode='markers',
                marker=dict(color=color, size=8),
                name=group_name, customdata=c_df[['filename']],
                hovertemplate="<b>%{customdata[0]}</b><br>" + metric_name + ": %{x:.4f}<extra></extra>"
            ))
        
        # K > 0 일 때만 중심점(X) 표시
        if centroids is not None:
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=[0] * len(centroids), mode='markers',
                marker=dict(color='red', symbol='x', size=15, line=dict(width=2, color='darkred')), 
                name='중심점', hoverinfo='skip'
            ))
        
        fig.add_vrect(
            x0=selected_range[0], x1=selected_range[1],
            fillcolor="green", opacity=0.15, layer="below", line_width=2, line_color="#4CAF50"
        )
        
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=f"{metric_name} 측정 수치")
        
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True, key=f"chart_{metric_name}")
        
        # 한눈에 보는 데이터 표 (리스트 형태)
        with st.expander(f"선택된 데이터 표로 보기 ({len(selected_df)}개)", expanded=False):
            display_df = selected_df[['filename', 'cluster', 'value']].rename(columns={'value': metric_name}).sort_values(by=metric_name)
            if selected_k == 0:
                display_df = display_df.drop(columns=['cluster'])
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
        # 누락(측정 실패) 데이터 안내 및 표
        if not missing_df.empty:
            st.warning(f"{len(missing_df)}개의 폰트는 폴리곤 인식 실패 등으로 인해 {metric_name} 값이 측정되지 않았습니다.")
            with st.expander("측정 실패 폰트 목록 보기", expanded=False):
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            
    with col2:
        st.subheader("선택된 이미지 확인")
        if event and len(event.selection.points) > 0 and "customdata" in event.selection.points[0]:
            selected_filename = event.selection.points[0]["customdata"][0]
            orig_fname = get_orig_fname(selected_filename)
            actual_crop = f"{selected_filename}_R_measured"
            
            orig_img = get_image_from_github("Seg_Mask_Labeling", orig_fname)
            if not orig_img: orig_img = get_image_from_github("Seg_RGB", orig_fname)
            crop_img = get_image_from_github("R_result", actual_crop)
            
            c1, c2 = st.columns(2)
            if orig_img: c1.image(orig_img, caption="원본", use_container_width=True)
            if crop_img: c2.image(crop_img, caption="측정 시각화(R)", use_container_width=True)
            st.divider()
            
        with st.expander(f"데이터 카드 모두 보기 ({len(selected_df)}개)", expanded=False):
            card_container = st.container(height=550)
            for idx, row in selected_df.iterrows():
                bg_color = DEFAULT_COLOR if selected_k == 0 else COLOR_MAP.get(row['cluster'], '#555')
                font_id = row['filename'].split(' ')[0] 
                group_label = "" if selected_k == 0 else f" (그룹 {row['cluster']})"
                
                orig_fname = get_orig_fname(row['filename'])
                actual_crop = f"{row['filename']}_R_measured"
                
                with card_container.container(border=True):
                    st.markdown(f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;'><b>{font_id}{group_label}</b></div>", unsafe_allow_html=True)
                    img_c1, img_c2 = st.columns(2)
                    
                    orig_img = get_image_from_github("Seg_Mask_Labeling", orig_fname)
                    if not orig_img: orig_img = get_image_from_github("Seg_RGB", orig_fname)
                    if orig_img: img_c1.image(orig_img, caption="원본", use_container_width=True)
                    else: img_c1.caption("원본 없음")
                    
                    crop_img = get_image_from_github("R_result", actual_crop)
                    if crop_img: img_c2.image(crop_img, caption="결과", use_container_width=True)
                    else: img_c2.caption("결과 없음")
                    
                    st.caption(f"{metric_name}: {row['value']:.4f}")

# ==========================================
# 3. 사이드바 및 메인 실행
# ==========================================
st.sidebar.title("M0~M5 분석 대시보드")

MENUS = {
    "M0": "상대적 유격 (R 폰트 바운딩박스 높이 기준)",
    "M1": "상대적 유격 (추출된 모든 글자 평균 높이 기준)",
    "M2": "상대적 유격 (추출된 모든 글자 면적 스케일 기준)",
    "M3": "상대적 유격 (랜드마크 Centroid Size 기준)",
    "M4": "Canonical Space 매핑 후 절대 유격 거리",
    "M5": "국소 두께(기둥) 대비 유격 비율"
}

if "menu" not in st.session_state:
    st.session_state.menu = "M0"

def change_menu(new_menu):
    st.session_state.menu = new_menu

for m_key in MENUS.keys():
    btn_type = "primary" if st.session_state.menu == m_key else "secondary"
    st.sidebar.button(f"{m_key} 지표 분석", type=btn_type, use_container_width=True, on_click=change_menu, args=(m_key,))

# 윈도우 환경 호환성 및 URL 규칙을 위해 os.path.join 대신 슬래시(/) 문자열로 직접 지정
file_path = "R_result/M0_M5_Analysis.txt"
df = load_and_parse_m_metrics(file_path)

if df is not None:
    current_menu = st.session_state.menu
    render_metric_page(current_menu, MENUS[current_menu], df)
else:
    st.error(f"분석 결과 파일을 불러올 수 없습니다. 경로를 확인해주세요: `{file_path}`")
