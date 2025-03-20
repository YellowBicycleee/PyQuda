import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

# 定义颜色
colors = {
    '6': 'rgb(255,0,0)',    # 红色
    '9': 'rgb(0,0,255)',    # 蓝色
    '12': 'rgb(0,255,0)'    # 绿色
}

# 读取CSV文件
def load_data(color):
    file1 = f'data/su{color}_start_from_lambda_5.csv'
    file2 = f'data/su{color}_single_lambda.csv'
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        return None, None
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1, df2

# 创建2D图表
def create_2d_plot(dfs_dict, axis, value, point_size):
    fig = go.Figure()
    
    for color_key in dfs_dict.keys():
        df1, df2 = dfs_dict[color_key]
        
        if axis == 'traj':
            if 0 <= int(value) < df1.shape[0]:
                y1 = df1.columns.astype(float).values
                z1 = df1.iloc[int(value)].values
                fig.add_trace(go.Scatter(
                    x=y1, y=z1, 
                    mode='lines+markers',
                    marker=dict(size=point_size, color=colors[color_key]),
                    line=dict(color=colors[color_key], dash='dash'),  # 虚线
                    name=f'SU({color_key}) strong coupling'
                ))
            
            if 0 <= int(value) < df2.shape[0]:
                y2 = df2.columns.astype(float).values
                z2 = df2.iloc[int(value)].values
                fig.add_trace(go.Scatter(
                    x=y2, y=z2, 
                    mode='lines+markers',
                    marker=dict(size=point_size, color=colors[color_key]),
                    line=dict(color=colors[color_key]),  # 实线
                    name=f'SU({color_key}) weak coupling'
                ))
            
        else:  # axis == 'lambda'
            if str(value) in df1.columns:
                x1 = np.arange(df1.shape[0])
                z1 = df1[str(value)].values
                fig.add_trace(go.Scatter(
                    x=x1, y=z1, 
                    mode='lines+markers',
                    marker=dict(size=point_size, color=colors[color_key]),
                    line=dict(color=colors[color_key], dash='dash'),  # 虚线
                    name=f'SU({color_key}) strong coupling'
                ))
            
            if str(value) in df2.columns:
                x2 = np.arange(df2.shape[0])
                z2 = df2[str(value)].values
                fig.add_trace(go.Scatter(
                    x=x2, y=z2, 
                    mode='lines+markers',
                    marker=dict(size=point_size, color=colors[color_key]),
                    line=dict(color=colors[color_key]),  # 实线
                    name=f'SU({color_key}) weak coupling'
                ))

    title_suffix = f'traj={value}' if axis == 'traj' else f'lambda={value}'
    fig.update_layout(
        title=f'2D Plot for {title_suffix}',
        xaxis_title='lambda' if axis == 'traj' else 'traj',
        yaxis_title='plaquette',
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    if axis == 'traj':
        fig.update_layout(xaxis=dict(range=[1.3, 1.6]))
    
    return fig

def main():
    st.title("2D Plot Comparison")
    
    # 加载所有可用的数据
    dfs_dict = {}
    for color in [6, 9, 12]:  # 检查这些特定的color值
        df1, df2 = load_data(str(color))
        if df1 is not None and df2 is not None:
            dfs_dict[str(color)] = (df1, df2)
    
    if not dfs_dict:
        st.error("没有找到可用的数据文件！")
        return
    
    st.write(f"找到以下SU(N)的数据: {', '.join(f'SU({c})' for c in dfs_dict.keys())}")
    
    # 添加滑块控制描点大小
    point_size = st.slider('选择2D图点的大小', min_value=1, max_value=20, value=5)
    
    # 添加选择框控制显示的二维图轴
    axis = st.selectbox('选择2D图显示轴', ['traj', 'lambda'])
    
    # 根据所有数据确定可用的范围
    if axis == 'traj':
        max_traj = min([min(df1.shape[0], df2.shape[0]) for df1, df2 in dfs_dict.values()]) - 1
        value = st.slider(f'选择 {axis}', min_value=0, max_value=max_traj, value=max_traj)
    else:
        lambda_sets = []
        for df1, df2 in dfs_dict.values():
            lambda_options1 = df1.columns.astype(float).values
            lambda_options2 = df2.columns.astype(float).values
            lambda_sets.append(set(lambda_options1) & set(lambda_options2))
        common_lambdas = sorted(list(set.intersection(*lambda_sets)))
        value = st.selectbox(f'选择 {axis}', common_lambdas)
    
    # 显示2D图
    st.write("### 2D Plot")
    fig2d = create_2d_plot(dfs_dict, axis, value, point_size)
    st.plotly_chart(fig2d)

if __name__ == '__main__':
    main()
