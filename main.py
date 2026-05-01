import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from linearRegression import LinearRegression
from frequencyResponse import FrequencyResponse
from modelChat import ModelChat

def save_uploaded_file(
        uploaded_file: any,
        path: str
    ):
    """
    保存上传的文件到指定路径
    """
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def render_tab1(
        script_path: str,
        data1_path: str,
        Re: float,
        R1: float,
        L: float
    ):
    """
    第一部分：阻尼振荡
        提供两种数据输入方式：文件上传、手动输入
        处理数据，计算衰减系数，显示结果和图表
    """
    st.header("阻尼振荡数据处理 (线性拟合)")
    input_method_1 = st.radio("选择数据输入方式", ["上传csv文件", "手动输入数据"], key="r1")
    has_data1 = False
    
    if input_method_1 == "上传csv文件":
        file1 = st.file_uploader("上传csv文件", type=["csv"], key="f1")
        if file1 is not None:
            save_uploaded_file(file1, data1_path)
            has_data1 = True
        else:
            st.warning("请勿上传空文件，确保文件格式正确，且包含两列数据。")
    else:
        st.info("请在下方输入数据。")
        df1 = pd.DataFrame([{"t / μs": 0.0, "Vc / V": 0.0} for _ in range(5)])
        edited_df1 = st.data_editor(df1, num_rows="dynamic", key="d1")
        edited_df1.to_csv(data1_path, index=False)
        has_data1 = True
            
    if has_data1 and st.button("处理阻尼振荡数据"):
        try:
            lr = LinearRegression(data1_path)
            lr.fit()
            lr.calc_confidence_band()
            
            st.subheader("处理结果")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("拟合直线", f"$\\ln(V_c) = {lr.k:.4f} t + {lr.b:.4f}$")
                st.metric("相关系数 $r$", f"${lr.r:.4f}$")
            with col2:
                beta_theory = (Re + R1) / (2 * L)
                beta_measured = - 10**6 * lr.k
                st.metric("理论衰减常数", f"${beta_theory:.4f} s^{{-1}}$")
                st.metric("实测衰减常数", f"${beta_measured:.4f} s^{{-1}}$")
            
            img_path = os.path.join(script_path, "result", "images", "linear_regression_temp.png")
            lr.plot(img_path)
            st.image(img_path, caption="线性拟合图表")

            # 保存结果到本地
            result_path = os.path.join(script_path, "result", "result_1.md")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("### 阻尼振荡数据处理结果\n\n")
                f.write(f"- **电源内阻值 $R_e$**: ${Re:.2f} \\Omega$\n")
                f.write(f"- **可变电阻值 $R_1$**: ${R1:.2f} \\Omega$\n")
                f.write(f"- **电感值 $L$**：${1000 * L:.2f} \\mathrm{{mH}}$\n")
                f.write(f"- **拟合直线**: $\\ln(V_c) = {lr.k:.4f} t + {lr.b:.4f}$\n")
                f.write(f"- **相关系数 $r$**: ${lr.r:.4f}$\n")
                f.write(f"- **理论衰减常数**: ${beta_theory:.4f} s^{{-1}}$\n")
                f.write(f"- **实测衰减常数**: ${beta_measured:.4f} s^{{-1}}$\n")
            st.success(f"结果已保存到 {result_path}")
        except Exception as e:
            st.error(f"处理出错: {e}")

def render_tab2(
        script_path: str,
        data2_path: str,
        Re: float,
        R1: float,
        R2: float,
        L: float,
        V_input: float
    ):
    """
    第二部分：频率响应
        提供两种数据输入方式：文件上传、手动输入
        计算共振频率、3dB带宽、品质因数、寄生电阻、修正衰减系数等，并显示结果
    """
    st.header("频率响应数据处理")
    input_method_2 = st.radio("选择数据输入方式", ["上传csv文件", "手动输入数据"], key="r2")
    has_data2 = False
    
    if input_method_2 == "上传csv文件":
        file2 = st.file_uploader("上传csv文件", type=["csv"], key="f2")
        if file2 is not None:
            save_uploaded_file(file2, data2_path)
            has_data2 = True
        else:
            st.warning("请勿上传空文件，确保文件格式正确，且包含二或三列数据。第一列为频率f，第二列为电压峰峰值Vpp，第三列为相位差。")
    else:
        st.info("请在下方输入数据。")
        df2 = pd.DataFrame([{"f / kHz": 0.0, "Vpp / V": 0.0, "φ / °": 0.0} for _ in range(5)])
        edited_df2 = st.data_editor(df2, num_rows="dynamic", key="d2")
        edited_df2.to_csv(data2_path, index=False)
        has_data2 = True
            
    if has_data2 and st.button("处理频率响应数据"):
        try:
            fr = FrequencyResponse(data2_path)
            f0, vpp_max, f1, f2, bandwidth, Q = fr.report()
            
            st.subheader("处理结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("共振频率 $f_0$", f"${f0:.2f} \\mathrm{{kHz}}$")
                st.metric("峰-峰值最大值", f"${vpp_max:.2f} \\mathrm{{V}}$")
                st.metric("品质系数 $Q$", f"${Q:.2f}$")
            with col2:
                st.metric("3dB带宽下限 $f_1$", f"${f1:.2f} \\mathrm{{kHz}}$")
                st.metric("3dB带宽上限 $f_2$", f"${f2:.2f} \\mathrm{{kHz}}$")
                st.metric("3dB带宽", f"${bandwidth:.2f} \\mathrm{{kHz}}$")
            with col3:
                R_parasitic = R2 * (V_input / vpp_max - 1)
                st.metric("寄生电阻", f"${R_parasitic:.2f} \\mathrm{{Ω}}$")
                beta_corrected =  (Re + R1 + R_parasitic) / (2 * L)
                st.metric("修正衰减常数", f"${beta_corrected:.4f} s^{{-1}}$")

            # 保存结果到本地
            result_path = os.path.join(script_path, "result", "result_2.md")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("### 频率响应数据处理结果\n\n")
                f.write(f"- **电源内阻值 $R_e$**: ${Re:.2f} \\Omega$\n")
                f.write(f"- **可变电阻值 $R_2$**: ${R2:.2f} \\Omega$\n")
                f.write(f"- **电感值 $L$**：${1000 * L:.2f} \\mathrm{{mH}}$\n")
                f.write(f"- **谐振时输入电压峰峰值 $V_{{\\mathrm{{input}}}}$** ：${V_input:.4f} \\mathrm{{V}}$\n")
                f.write(f"- **共振频率 $f_0$**: ${f0:.2f} \\mathrm{{kHz}}$\n")
                f.write(f"- **峰-峰值最大值**: ${vpp_max:.2f} \\mathrm{{V}}$\n")
                f.write(f"- **品质系数 $Q$**: ${Q:.2f}$\n")
                f.write(f"- **3dB带宽下限 $f_1$**: ${f1:.2f} \\mathrm{{kHz}}$\n")
                f.write(f"- **3dB带宽上限 $f_2$**: ${f2:.2f} \\mathrm{{kHz}}$\n")
                f.write(f"- **3dB带宽**: ${bandwidth:.2f} \\mathrm{{kHz}}$\n")
                f.write(f"- **寄生电阻**: ${R_parasitic:.2f} \\mathrm{{Ω}}$\n")
                f.write(f"- **修正衰减常数**: ${beta_corrected:.4f} s^{{-1}}$\n")
            st.success(f"结果已保存到 {result_path}")
        except Exception as e:
            st.error(f"处理出错: {e}")

def render_tab3(
        script_path: str,
        data1_path: str,
        data2_path: str,
        default_model_config_path: str
    ):
    """
    第三部分：报告生成
        读取前两部分处理结果，调用大语言模型生成结果分析
    """
    st.header("实验报告生成")
    st.info("请确保前两部分的数据处理已完成，并且结果已保存。将API密钥、基础URL和模型名称填入配置文件中，并输入提示词，即可一键生成结果分析。")

    with open(default_model_config_path, "r", encoding="utf-8") as f:
        default_config = json.load(f)[0]
    api_key = st.text_input("API密钥：`api_key`", value=default_config["api_key"])
    base_url = st.text_input("基础URL：`base_url`", value=default_config["base_url"])
    model_name = st.text_input("模型名称：`model_name`", value=default_config["model"])
    system_prompt = st.text_area("系统提示词：`system_prompt`", value=default_config["system_prompt"])
    report_prompt = st.text_area("报告提示词：`report_prompt`", value=default_config["report_prompt"])

    if st.button("生成实验报告"):
        if not os.path.exists(data1_path) or not os.path.exists(data2_path):
            st.error("请先完成前两部分的数据处理，并确保结果已保存。")
            return
        try:
            with open(os.path.join(script_path, "result", "result_1.md"), "r", encoding="utf-8") as f:
                result_1 = f.read()
            with open(os.path.join(script_path, "result", "result_2.md"), "r", encoding="utf-8") as f:
                result_2 = f.read()
            result = result_1 + "\n\n" + result_2
            chat = ModelChat(api_key=api_key, base_url=base_url, model=model_name)
            report = chat.report(
                system_prompt=system_prompt,
                report_prompt=report_prompt,
                result=result
            )
            report_path = os.path.join(script_path, "result", "report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            st.success(f"报告已保存到 {report_path}")
            st.markdown(report)
        except Exception as e:
            st.error(f"生成报告出错: {e}")

def main():
    # 标题
    st.set_page_config(page_title="RLC实验数据处理", layout="wide")
    st.title("RLC电路实验阻尼振荡与频率响应数据处理")

    # 侧边栏
    st.sidebar.header("实验参数设置")
    Re = st.sidebar.number_input("电源内阻值 $R_e / \\Omega$", value=50.0, step=0.1)
    R1 = st.sidebar.number_input("阻尼振动实验可变电阻值 $R_1 / \\Omega$", value=100.0, step=0.1)
    R2 = st.sidebar.number_input("受迫振动实验可变电阻值 $R_2 / \\Omega$", value=500.0, step=0.1)
    C = st.sidebar.number_input("电容值 $C / \\mathrm{nF}$", value=10.0, step=0.1) * 1e-9
    L = st.sidebar.number_input("电感值 $L / \\mathrm{mH}$", value=20.0, step=0.1) * 1e-3
    V_input = st.sidebar.number_input("谐振时输入电压峰峰值 $V_{\\mathrm{input}} / \\mathrm{V}$", value=5.0000, format="%.4f")

    # 数据与路径建立
    script_path = os.path.dirname(os.path.abspath(__file__))
    data1_path = os.path.join(script_path, "data", "data1_temp.csv")
    data2_path = os.path.join(script_path, "data", "data2_temp.csv")
    default_model_config_path = os.path.join(script_path, "defaultModelConfig.json")
    os.makedirs(os.path.join(script_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(script_path, "result/images"), exist_ok=True)

    # 主界面
    st.markdown("""
        **使用说明：**
        
        请先在侧边栏设置实验参数，包括：
        
        - 电源内阻值 $R_e / \\Omega$
        - 阻尼振动实验可变电阻值 $R_1 / \\Omega$
        - 受迫振动实验可变电阻值 $R_2 / \\Omega$
        - 电容值 $C / \\mathrm{nF}$
        - 电感值 $L / \\mathrm{mH}$
        - 谐振时输入电压峰峰值 $V_{\\mathrm{input}} / \\mathrm{V}$
        
        然后在下方输入数据（可选择上传csv文件或直接在表格中输入数据），一键处理数据，查看结果，并生成实验报告。

    """)
    tab1, tab2, tab3 = st.tabs(["第一部分：阻尼振荡", "第二部分：频率响应", "第三部分：报告生成"])
    with tab1:
        render_tab1(script_path, data1_path, Re, R1, L)
    with tab2:
        render_tab2(script_path, data2_path, Re, R1, R2, L, V_input)
    with tab3:
        render_tab3(script_path, data1_path, data2_path, default_model_config_path)


if __name__ == "__main__":
    main()