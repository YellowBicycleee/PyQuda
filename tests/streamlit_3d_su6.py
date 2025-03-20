import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

# 定义颜色
RED = 'rgb(255,0,0)'  # 红色
DARK_RED = 'rgb(128,0,0)'    # 深红
LIGHT_RED = 'rgb(255,182,193)'  # 浅红

def load_and_process_data():
    # 读取文件
    file1 = 'data_32x32/su6_from_lambda_5_full.csv'
    file2 = 'data_32x32/su6_single_lambda_full.csv'
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        return None, None
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # 处理数据：取每列最后100行的平均值
    mean_values1 = df1.iloc[-10:].mean()
    mean_values2 = df2.iloc[-10:].mean()
    
    return mean_values1, mean_values2

def create_plot(mean_values1, mean_values2, point_size):
    fig = go.Figure()
    
    # 添加从lambda 5开始的数据（使用深红色实线）
    x1 = mean_values1.index.astype(float)
    y1 = mean_values1.values
    fig.add_trace(go.Scatter(
        x=x1, y=y1,
        mode='lines+markers',
        marker=dict(size=point_size, color=DARK_RED),
        line=dict(color=DARK_RED),  # 改为深红色实线
        name='SU(6) strong coupling'
    ))
    
    # 添加single lambda的数据（使用浅红色实线）
    x2 = mean_values2.index.astype(float)
    y2 = mean_values2.values
    fig.add_trace(go.Scatter(
        x=x2, y=y2,
        mode='lines+markers',
        marker=dict(size=point_size, color=LIGHT_RED),
        line=dict(color=LIGHT_RED),  # 改为浅红色实线
        name='SU(6) weak coupling'
    ))
    
    # 更新图表布局
    fig.update_layout(
        title='SU(6) Plaquette vs Lambda',
        xaxis_title='lambda',
        yaxis_title='plaquette',
        xaxis=dict(range=[0, 5]),
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def main():
    st.title("SU(6) Plaquette Analysis")
    
    # 加载和处理数据
    mean_values1, mean_values2 = load_and_process_data()
    
    if mean_values1 is None or mean_values2 is None:
        st.error("没有找到数据文件！")
        return
    
    # 添加滑块控制描点大小
    point_size = st.slider('选择点的大小', min_value=1, max_value=20, value=5)
    
    # 显示图表
    st.write("### Plaquette vs Lambda")
    fig = create_plot(mean_values1, mean_values2, point_size)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
