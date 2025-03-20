import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

def calculate_grouped_averages(input_file, output_file, line, subline):
    """
    读取CSV文件，计算每列最后line行数据，每subline行的平均值
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    line: 要处理的最后几行数据
    subline: 每几行计算一次平均值
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查line是否超过文件行数
        total_rows = len(df)
        if line > total_rows:
            raise ValueError(f"指定的行数({line})超过了文件总行数({total_rows})")
            
        # 获取最后line行数据
        last_rows = df.tail(line)
        
        # 计算需要多少组
        num_groups = (line + subline - 1) // subline
        group_averages = []
        
        # 对每subline行计算一次平均值
        for i in range(num_groups):
            start_idx = -(min(line, total_rows)) + i * subline
            end_idx = start_idx + subline
            group_data = last_rows.iloc[start_idx:end_idx]
            if not group_data.empty:
                group_avg = group_data.mean()
                group_averages.append(group_avg)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(group_averages, columns=df.columns)
        
        # 保存结果到新的CSV文件
        result_df.to_csv(output_file, index=False)
        print(f"处理完成！结果已保存到: {output_file}")
        
        # 读取结果CSV文件并绘制图形
        result_df.plot()
        plt.title('Grouped Averages')
        plt.xlabel('Group Index')
        plt.ylabel('Average Value')
        plt.legend(title='Columns')
        plt.show()
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='处理CSV文件并计算分组平均值')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件路径')
    parser.add_argument('--line', type=int, required=True, help='要处理的最后几行数据')
    parser.add_argument('--subline', type=int, required=True, help='每几行计算一次平均值')
    
    args = parser.parse_args()
    
    calculate_grouped_averages(args.input, args.output, args.line, args.subline)

if __name__ == "__main__":
    main()
