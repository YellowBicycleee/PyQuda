import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

# 读取CSV文件
def load_data(color):
    file1 = f'data/su{color}_start_from_lambda_5.csv'
    file2 = f'data/su{color}_single_lambda.csv'
    
    # 检查文件是否存在
    if not os.path.exists(file1) or not os.path.exists(file2):
        return None, None
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1, df2

# 创建3D图表
def create_3d_surface(df, x_max, title):
    y = df.columns.astype(float).values  # lambda
    x = np.arange(df.shape[0])  # traj
    x = x[x <= x_max]  # 限制 x 轴的上限
    df = df.iloc[:len(x)]  # 限制数据范围
    X, Y = np.meshgrid(x, y)
    Z = df.values.T  # 转置使得数据对齐

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='traj',
            yaxis_title='lambda',
            zaxis_title='plaquette',
            zaxis=dict(range=[0, 1]),  # 限制 z 轴范围在 [0, 1]
            yaxis=dict(range=[0, 5])   # 限制 y 轴范围在 [0, 5]
        )
    )
    return fig

# 创建2D图表
def create_2d_plot(df1, df2, axis, value, point_size, same_color=True):
    fig = go.Figure()
    
    # 从文件路径中提取文件名
    file1 = 'data/su9_start_from_lambda_5.csv'.split('/')[-1]
    file2 = 'data/su9_single_lambda.csv'.split('/')[-1]
    
    # 设置颜色
    color1 = 'blue' if same_color else 'blue'
    color2 = 'blue' if same_color else 'red'

    if axis == 'traj':
        # CSV1
        y1 = df1.columns.astype(float).values
        z1 = df1.iloc[int(value)].values
        fig.add_trace(go.Scatter(
            x=y1, y=z1, mode='lines+markers',
            marker=dict(size=point_size, color=color1),
            line=dict(color=color1),
            name=file1
        ))
        
        # CSV2
        y2 = df2.columns.astype(float).values
        z2 = df2.iloc[int(value)].values
        fig.add_trace(go.Scatter(
            x=y2, y=z2, mode='lines+markers',
            marker=dict(size=point_size, color=color2),
            line=dict(color=color2),
            name=file2
        ))
        
        fig.update_layout(
            title=f'2D Plot for traj={value}',
            xaxis_title='lambda',
            yaxis_title='plaquette',
            xaxis=dict(range=[1.3, 1.5]),
            yaxis=dict(range=[0, 1]),
            legend=dict(
                orientation="h",  # 水平放置图例
                yanchor="bottom",
                y=-0.3,  # 将图例放在图表下方
                xanchor="center",
                x=0.5
            )
        )
        
    else:
        # 类似的修改应用于 lambda 轴的情况
        x1 = np.arange(df1.shape[0])
        z1 = df1.loc[:, str(value)].values
        fig.add_trace(go.Scatter(
            x=x1, y=z1, mode='lines+markers',
            marker=dict(size=point_size, color=color1),
            line=dict(color=color1),
            name=file1
        ))
        
        x2 = np.arange(df2.shape[0])
        z2 = df2.loc[:, str(value)].values
        fig.add_trace(go.Scatter(
            x=x2, y=z2, mode='lines+markers',
            marker=dict(size=point_size, color=color2),
            line=dict(color=color2),
            name=file2
        ))
        
        fig.update_layout(
            title=f'2D Plot for lambda={value}',
            xaxis_title='traj',
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
    
    return fig

def main():
    st.title("3D Surface Plot from CSV Data")
    st.write("This app shows the 3D surface plot and 2D plot of two CSV files.")
    
    # 添加颜色选择
    # 首先检查可用的颜色
    available_colors = []
    for color in range(2, 13):
        file1 = f'data/su{color}_start_from_lambda_5.csv'
        file2 = f'data/su{color}_single_lambda.csv'
        if os.path.exists(file1) and os.path.exists(file2):
            available_colors.append(color)
    
    if not available_colors:
        st.error("no available data file!")
        return
    
    color = st.selectbox('select your color', available_colors)
    
    df1, df2 = load_data(color)
    if df1 is None or df2 is None:
        st.error(f"can not load SU({color}) data file!")
        return
    
    # 添加滑块控制描点大小
    point_size = st.slider('select 2D plot point size', min_value=1, max_value=20, value=5)
    
    # 添加选择框控制显示的二维图轴
    axis = st.selectbox('select 2D plot axis', ['traj', 'lambda'])
    
    if axis == 'traj':
        max_traj = min(df1.shape[0], df2.shape[0]) - 1
        value = st.slider(f'select {axis}', min_value=0, max_value=max_traj, value=max_traj, key='traj_select')
    else:
        lambda_options1 = df1.columns.astype(float).values.tolist()
        lambda_options2 = df2.columns.astype(float).values.tolist()
        lambda_min = max(min(lambda_options1), min(lambda_options2))
        lambda_max = min(max(lambda_options1), max(lambda_options2))
        lambda_options = [lam for lam in lambda_options1 if lambda_min <= lam <= lambda_max]
        value = st.selectbox(f'select {axis}', lambda_options, key='lambda_select')
    
    # 添加颜色选择
    same_color = st.checkbox('use same color', value=True)

    st.write("### 2D Plot")
    fig2d = create_2d_plot(df1, df2, axis, value, point_size, same_color)
    st.plotly_chart(fig2d)
    
    # 添加滑块控制 x 轴上限
    x_max1 = st.slider('select CSV1 x axis upper limit (traj)', min_value=0, max_value=df1.shape[0]-1, value=df1.shape[0]-1, key='x_max1')
    x_max2 = st.slider('select CSV2 x axis upper limit (traj)', min_value=0, max_value=df2.shape[0]-1, value=df2.shape[0]-1, key='x_max2')
    
    # 创建并展示CSV1的3D图
    fig3d_1 = create_3d_surface(df1, x_max1, 'CSV1 的 3D Surface Plot')
    st.plotly_chart(fig3d_1)
    
    # 创建并展示CSV2的3D图
    fig3d_2 = create_3d_surface(df2, x_max2, 'CSV2 的 3D Surface Plot')
    st.plotly_chart(fig3d_2)

if __name__ == '__main__':
    main()
