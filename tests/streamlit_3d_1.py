import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# 读取CSV文件
def load_data():
    # df1 = pd.read_csv('data/su6_single_beta.csv')
    # df2 = pd.read_csv('data/su6_begin_const0.6_traj500_others_2000.csv')
    df1 = pd.read_csv('data/su9_start_from_lambda_5.csv')
    df2 = pd.read_csv('data/su9_single_lambda.csv')
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
def create_2d_plot(df1, df2, axis, value, point_size):
    fig = go.Figure()
    color = 'blue'  # 定义统一的颜色

    if axis == 'traj':
        # CSV1
        y1 = df1.columns.astype(float).values  # lambda
        z1 = df1.iloc[int(value)].values  # 选择特定的 traj
        fig.add_trace(go.Scatter(
            x=y1, y=z1, mode='lines+markers',
            marker=dict(size=point_size, color=color),
            line=dict(color=color),
            name='CSV1'
        ))
        
        # CSV2
        y2 = df2.columns.astype(float).values  # lambda
        z2 = df2.iloc[int(value)].values  # 选择特定的 traj
        fig.add_trace(go.Scatter(
            x=y2, y=z2, mode='lines+markers',
            marker=dict(size=point_size, color=color),
            line=dict(color=color),
            name='CSV2'
        ))
        
        fig.update_layout(
            title=f'2D Plot for traj={value}',
            xaxis_title='lambda',
            yaxis_title='plaquette',
            xaxis=dict(range=[1.4, 1.5]),
            yaxis=dict(range=[0, 1])  # 限制 y 轴范围在 [0, 1]
        )
        
    else:
        # CSV1
        x1 = np.arange(df1.shape[0])  # traj
        z1 = df1.loc[:, str(value)].values  # 选择特定的 lambda
        fig.add_trace(go.Scatter(
            x=x1, y=z1, mode='lines+markers',
            marker=dict(size=point_size),
            name='CSV1'
        ))
        
        # CSV2
        x2 = np.arange(df2.shape[0])  # traj
        z2 = df2.loc[:, str(value)].values  # 选择特定的 lambda
        fig.add_trace(go.Scatter(
            x=x2, y=z2, mode='lines+markers',
            marker=dict(size=point_size),
            name='CSV2'
        ))
        
        fig.update_layout(
            title=f'2D Plot for lambda={value}',
            xaxis_title='traj',
            yaxis_title='plaquette',
            yaxis=dict(range=[0, 1])  # 限制 y 轴范围在 [0, 1]
        )
        
    return fig

def main():
    st.title("3D Surface Plot from CSV Data")
    st.write("此应用同时展示两个CSV文件的3D表面图和二维图。")
    
    df1, df2 = load_data()
    
    # 添加滑块控制 x 轴上限
    x_max1 = st.slider('选择 CSV1 的 x 轴上限 (traj)', min_value=0, max_value=df1.shape[0]-1, value=df1.shape[0]-1, key='x_max1')
    x_max2 = st.slider('选择 CSV2 的x 轴上限 (traj)', min_value=0, max_value=df2.shape[0]-1, value=df2.shape[0]-1, key='x_max2')
    
    # 创建并展示CSV1的3D图
    fig3d_1 = create_3d_surface(df1, x_max1, 'CSV1 的 3D Surface Plot')
    st.plotly_chart(fig3d_1)
    
    # 创建并展示CSV2的3D图
    fig3d_2 = create_3d_surface(df2, x_max2, 'CSV2 的 3D Surface Plot')
    st.plotly_chart(fig3d_2)
    
    # 添加滑块控制描点大小
    point_size = st.slider('选择二维图的描点大小', min_value=1, max_value=20, value=5)
    
    # 添加选择框控制显示的二维图轴
    axis = st.selectbox('选择二维图的轴', ['traj', 'lambda'])
    
    if axis == 'traj':
        max_traj = min(df1.shape[0], df2.shape[0]) - 1
        value = st.slider(f'选择 {axis}', min_value=0, max_value=max_traj, value=max_traj, key='traj_select')
    else:
        lambda_options1 = df1.columns.astype(float).values.tolist()
        lambda_options2 = df2.columns.astype(float).values.tolist()
        # 取两个lambda的交集或统一范围
        lambda_min = max(min(lambda_options1), min(lambda_options2))
        lambda_max = min(max(lambda_options1), max(lambda_options2))
        lambda_options = [lam for lam in lambda_options1 if lambda_min <= lam <= lambda_max]
        value = st.selectbox(f'选择 {axis}', lambda_options, key='lambda_select')
    
    st.write("### 选择的二维图")
    fig2d = create_2d_plot(df1, df2, axis, value, point_size)
    st.plotly_chart(fig2d)

if __name__ == '__main__':
    main()
