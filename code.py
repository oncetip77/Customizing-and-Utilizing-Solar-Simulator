import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.layouts import Modelspace

def extract_geo_from_dxf(filename: str):
    """
    DXF 파일에서 기하학적 정보를 추출합니다.
    - 'WAFER' 레이어에서 웨이퍼 면적을 계산합니다.
    - 'METAL' 레이어에서 금속 전극 면적을 계산합니다.
    """
    try:
        doc = ezdxf.readfile(filename)
        msp = doc.modelspace()
    except IOError:
        print(f"오류: DXF 파일을 찾을 수 없습니다: {filename}")
        return None, None, None
    except ezdxf.DXFStructureError:
        print(f"오류: 유효하지 않은 DXF 파일입니다: {filename}")
        return None, None, None

    def get_area_from_layer(msp: Modelspace, layer_name: str) -> float:
        """지정된 레이어의 모든 닫힌 폴리라인 면적 합계를 계산합니다."""
        total_area_mm2 = 0
        polylines = msp.query(f'LWPOLYLINE[layer=="{layer_name}"]')
        for polyline in polylines:
            if polyline.is_closed:
                points = [(p[0], p[1]) for p in polyline.get_points()]
                # Shoelace formula로 면적 계산
                area = 0.5 * abs(sum(p1[0]*p2[1] - p2[0]*p1[1] for p1, p2 in zip(points, points[1:] + [points[0]])))
                total_area_mm2 += area
        return total_area_mm2

    # 면적 계산 (단위: mm^2)
    wafer_area_mm2 = get_area_from_layer(msp, 'WAFER')
    metal_area_mm2 = get_area_from_layer(msp, 'METAL')

    if wafer_area_mm2 == 0:
        print("경고: 'WAFER' 레이어에서 면적을 계산할 수 없습니다. 웨이퍼 면적을 0으로 처리합니다.")
        return 0, 0, 0

    # 단위 변환 (mm^2 -> cm^2)
    wafer_area_cm2 = wafer_area_mm2 / 100
    metal_area_cm2 = metal_area_mm2 / 100

    shading_ratio = metal_area_cm2 / wafer_area_cm2

    return wafer_area_cm2, metal_area_cm2, shading_ratio

def save_detailed_report(filename, params):
    """요청된 형식에 맞춰 상세 결과 리포트를 텍스트 파일로 저장합니다."""

    # 보고서용 파라미터 계산
    shading_ratio_report = params['shading_ratio_for_report']
    wafer_area = params['wafer_area_cm2']

    front_metal_area = wafer_area * shading_ratio_report

    # 광 손실을 적용한 실제 IV 파라미터 추정
    Jsc_real = params['Jsc_actual'] * (1 - shading_ratio_report)
    Jmp_real = params['Jmp'] * (1 - shading_ratio_report)
    Imp_real = Jmp_real * wafer_area / 1000
    Pmp_real = params['Vmp'] * Jmp_real
    Eff_real = Pmp_real / params['P_in']

    Rs_ohm_cm2 = params['Rs_total_ohm'] * wafer_area

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Cell Parameters\n")
        f.write(f"wafer area\t{wafer_area:.6f}\tcm2\n")
        f.write(f"front metal contact area\t{front_metal_area:.6f}\tcm2\t{shading_ratio_report*100:.6f}\t%\n")
        f.write(f"rear metal contact area\t0.000000\tcm2\t0.000000\t%\n")
        f.write("\n")

        f.write("IV Parameters\n")
        f.write(f"front JL(non-shaded regions)\t{params['Jsc_actual']:.6f}\tmA/cm2\n")
        f.write(f"front JL\t{Jsc_real:.6f}\tmA/cm2\n")
        f.write(f"Jsc\t{Jsc_real:.6f}\tmA/cm2\n")
        f.write(f"Voc\t{params['Voc']*1000:.6f}\tmV\n")
        f.write(f"FF\t{params['FF']*100:.6f}\t%\n")
        f.write(f"Efficiency\t{Eff_real*100:.6f}\t%\n")
        f.write(f"Vmp\t{params['Vmp']*1000:.6f}\tmV\n")
        f.write(f"Imp\t{Imp_real:.6f}\tA\n")
        f.write(f"Jmp\t{Jmp_real:.6f}\tmA/cm2\n")
        f.write(f"Rs\t{Rs_ohm_cm2:.6f}\tohm-cm2\n")
        f.write(f"Rs\t{params['Rs_total_ohm']*1000:.6f}\tmohm\n")
        f.write("\n")

        f.write("Power analysis at terminal voltage of\t0.000000\tmV\n")
        f.write("NOTE: 아래의 상세 손실 분석은 현재 모델에서 지원하지 않습니다.\n")
        f.write("power output\t{:.6f}\tmW/cm2\t{:.6f}\t%\n".format(Pmp_real, (Pmp_real/params['Jsc_ideal'])*100))
        f.write("front shading\t{:.6f}\tmW/cm2\t{:.6f}\t%\n".format(params['P_in'] * shading_ratio_report, (params['P_in'] * shading_ratio_report / params['Jsc_ideal'])*100))
        f.write("\n")

        f.write("JV Curve\n")
        f.write("V\tJ\tpower density\n")
        f.write("(mV)\t(mA/cm2)\t(mW/cm2)\n")
        j_range_real = params['J_range'] * (1 - shading_ratio_report)
        p_density_real = params['V_range'] * j_range_real
        indices_to_save = [0, len(params['V_range'])//2, params['idx_max'], len(params['V_range'])-1]
        for i in indices_to_save:
            v_val = params['V_range'][i] * 1000
            j_val = j_range_real[i]
            p_val = p_density_real[i]
            f.write(f"{v_val:.6f}\t{j_val:.6f}\t{p_val:.6f}\n")
        f.write("\n")

# ===== 입력값 설정 =====
# --- 1. DXF 파일에서 기하학적 정보 추출 ---
dxf_filename = "180_bb_1.4divided.dxf"  # 분석할 DXF 파일명을 입력하세요.
wafer_area_cm2, metal_area_cm2, shading_ratio_for_report = extract_geo_from_dxf(dxf_filename)

if wafer_area_cm2 is None:
    print("DXF 파일 처리 중 오류가 발생하여 시뮬레이션을 중단합니다.")
else:
    print(f"--- DXF 분석 결과 ---")
    print(f"웨이퍼 면적: {wafer_area_cm2:.3f} cm²")
    print(f"금속 전극 면적: {metal_area_cm2:.3f} cm²")
    print(f"차광 손실률: {shading_ratio_for_report*100:.3f} %")
    print("-" * 30)

    # --- 2. 물리/전기적 특성 수동 입력 ---
    # NOTE: 이 값들은 DXF에서 추출할 수 없으므로, 직접 입력해야 합니다.
    Jsc_ideal = 40.63
    Voc = 0.704
    n_ideality = 1.2
    temperature_K = 298
    Rs_total_ohm = 5.61e-3
    Rsh_ohm_cm2 = 5000.0

    # --- 물리 상수 ---
    k = 1.381e-23
    q = 1.602e-19

    # ===== 계산 실행 =====
    # 시뮬레이션은 항상 활성 영역(shading=0) 기준으로 실행
    shading_ratio_sim = 0.0

    Jph = Jsc_ideal * (1 - shading_ratio_sim)
    Vt = k * temperature_K / q
    J0 = Jsc_ideal / (np.exp(Voc / (n_ideality * Vt)) - 1)

    n_points = 200 # J-V 데이터 포인트 갯수
    V_range = np.linspace(0, Voc, n_points)
    J_range = np.zeros_like(V_range)

    for i, v in enumerate(V_range):
        J_guess = Jph
        for _ in range(10):
            I_guess_A = (J_guess / 1000) * wafer_area_cm2
            V_diode = v - I_guess_A * Rs_total_ohm
            if V_diode < 0:
                V_diode = 0
            J_shunt = (V_diode / Rsh_ohm_cm2) * 1000
            J_new = Jph - J0 * (np.exp(V_diode / (n_ideality * Vt)) - 1) - J_shunt
            J_guess = (J_guess + J_new) / 2
        J_range[i] = J_guess
    J_range[J_range < 0] = 0

    # (성능 지표 계산 및 결과 출력, 리포트 저장, 시각화)
    P_density = V_range * J_range
    idx_max = np.argmax(P_density)
    Vmp = V_range[idx_max]
    Jmp = J_range[idx_max]
    Pmp = P_density[idx_max]
    Jsc_actual = J_range[0]
    FF = (Vmp * Jmp) / (Voc * Jsc_actual)
    P_in = 100
    Efficiency = Pmp / P_in

    print("--- 활성 영역(Active Area) 성능 시뮬레이션 결과 ---")
    print(f"Jsc (Ideal)  = {Jsc_actual:.2f} mA/cm²")
    print(f"Voc         = {Voc*1000:.2f} mV")
    print(f"Vmp         = {Vmp*1000:.2f} mV")
    print(f"Jmp         = {Jmp:.2f} mA/cm²")
    print(f"Pmp         = {Pmp:.2f} mW/cm²")
    print(f"FF          = {FF*100:.2f} %")
    print(f"Efficiency  = {Efficiency*100:.2f} %")
    print("-" * 30)

    report_params = {
        'wafer_area_cm2': wafer_area_cm2, 'shading_ratio_for_report': shading_ratio_for_report,
        'Jsc_ideal': Jsc_ideal, 'Voc': Voc, 'FF': FF, 'P_in': P_in, 'Rs_total_ohm': Rs_total_ohm,
        'Jsc_actual': Jsc_actual, 'Vmp': Vmp, 'Jmp': Jmp,
        'V_range': V_range, 'J_range': J_range, 'idx_max': idx_max
    }
    report_filename = "simulation_summary_report.txt"
    save_detailed_report(report_filename, report_params)
    print(f"상세 결과 리포트 '{report_filename}' 파일 저장 완료.")

    plt.figure(figsize=(8, 6))
    plt.plot(V_range * 1000, J_range, label="J–V Curve (Active Area)", color="blue", linewidth=2)
    plt.axvline(Vmp * 1000, color="red", linestyle="--", label=f"Vmp = {Vmp*1000:.1f} mV")
    plt.axhline(Jmp, color="green", linestyle="--", label=f"Jmp = {Jmp:.2f} mA/cm²")
    plt.title("J-V Curve (Active Area)", fontsize=16)
    plt.xlabel("Voltage (mV)", fontsize=12)
    plt.ylabel("Current Density (mA/cm²)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
