import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from PIL import Image

st.set_page_config(page_title="C 폰트 클러스터링", layout="wide")
st.title("C 폰트 상단/하단 두께 비율 K-Means 클러스터링")

file_path = r"./C_result/C_basic_analysis.txt"
original_image_dir = r"./Seg_RGB"
crop_image_dir = r"./C_result" 

def get_image_path_with_ext(base_dir, file_name_without_ext):
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        full_path = os.path.join(base_dir, file_name_without_ext + ext)
        if os.path.exists(full_path):
            return full_path
    return None

@st.cache_data
def load_and_process_data(file_path):
    if not os.path.exists(file_path):
        return None, None
        
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    blocks = text_data.split("파일명:")[1:]
    data_list = []
    
    for block in blocks:
        filename = block.split('\n')[0].strip()
        top_match = re.search(r"상단 대비\s*([\d\.]+)\s*배", block)
        bottom_match = re.search(r"하단 대비\s*([\d\.]+)\s*배", block)
        
        if top_match and bottom_match:
            top_val = float(top_match.group(1))
            bottom_val = float(bottom_match.group(1))
            
            # 5.0 이상 이상치 제외
            if top_val < 5.0 and bottom_val < 5.0:
                data_list.append({
                    "filename": filename,
                    "top_ratio": top_val,
                    "bottom_ratio": bottom_val
                })
                
    if not data_list:
        return None, None
        
    df = pd.DataFrame(data_list)
    
    X = df[['top_ratio', 'bottom_ratio']].values
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    df['cluster'] = df['cluster'].astype(str)
    
    centroids = kmeans.cluster_centers_
    return df, centroids

df, centroids = load_and_process_data(file_path)

if df is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("k means 클러스터링 결과 (점 클릭)")
        
        fig = px.scatter(
            df, 
            x="top_ratio", 
            y="bottom_ratio", 
            color="cluster",
            hover_data=["filename"],
            labels={"top_ratio": "상단 대비 비율 (배)", "bottom_ratio": "하단 대비 비율 (배)", "cluster": "그룹"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(color='red', symbol='x', size=12),
            name='중심점',
            hoverinfo='skip'
        ))
        
        fig.update_layout(height=700)
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)

    with col2:
        st.subheader("선택된 이미지 확인")
        
        if event and len(event.selection.points) > 0:
            selected_point = event.selection.points[0]
            
            if "customdata" in selected_point:
                selected_filename = selected_point["customdata"][0]
                
                if '_crop' in selected_filename:
                    original_filename = selected_filename.split('_crop')[0] + '_crop'
                else:
                    original_filename = selected_filename
                
                st.markdown(f"**원본 파일:** `{original_filename}`")
                orig_img_path = get_image_path_with_ext(original_image_dir, original_filename)
                
                if orig_img_path:
                    image_orig = Image.open(orig_img_path)
                    st.image(image_orig, use_container_width=True)
                else:
                    st.warning(f"원본 이미지를 찾을 수 없습니다.\n({original_image_dir} 경로 확인)")

                st.divider()
                
                actual_crop_filename = f"{selected_filename}_C_stroke_basic"
                
                st.markdown(f"**크롭 파일:** `{actual_crop_filename}`")
                crop_img_path = get_image_path_with_ext(crop_image_dir, actual_crop_filename)
                
                if crop_img_path:
                    image_crop = Image.open(crop_img_path)
                    st.image(image_crop, use_container_width=True)
                else:
                    st.warning(f"크롭 이미지를 찾을 수 없습니다.\n({crop_image_dir} 경로 확인)")
                    
            else:
                st.info("중심점(빨간색 X)이 아닌 데이터 포인트를 클릭")
        else:
            st.info("왼쪽 그래프에서 데이터 포인트를 클릭")

else:
    st.error(f"'{file_path}' 파일을 읽을 수 없거나 유효한 데이터가 없습니다.")