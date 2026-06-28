import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime

def generate_pdf_report(run_session_info, results_dir, output_pdf_path):
    """
    Generates a 4-page PDF Runner Analysis Report.
    run_session_info: dict containing runnerName, date (datetime), fps, cameraCount, totalTime, avgVelocity, avgAcceleration, avgStepLength, note
    results_dir: directory path containing metrics.csv / angles.csv
    output_pdf_path: path to save the generated PDF
    """
    # 1. Locate CSV files
    metrics_path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(results_dir, "tracking_results", "run_metrics.csv")

    angles_path = os.path.join(results_dir, "angles.csv")
    if not os.path.exists(angles_path):
        angles_path = os.path.join(results_dir, "tracking_results", "joint_angles.csv")

    # Load DataFrames
    df_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    df_angles = pd.read_csv(angles_path) if os.path.exists(angles_path) else None

    # Handle time alignment for metrics
    fps = run_session_info.get("fps") or 60.0
    if df_metrics is not None:
        if "time_s" not in df_metrics.columns:
            if "absolute_frame" in df_metrics.columns:
                df_metrics["time_s"] = df_metrics["absolute_frame"] / fps
            elif "frame" in df_metrics.columns:
                df_metrics["time_s"] = df_metrics["frame"] / fps
        if "time_s" in df_metrics.columns:
            df_metrics = df_metrics.sort_values("time_s")

    # Handle time alignment for angles
    if df_angles is not None:
        if "time_s" not in df_angles.columns:
            if "absolute_frame" in df_angles.columns:
                df_angles["time_s"] = df_angles["absolute_frame"] / fps
            elif "frame" in df_angles.columns:
                df_angles["time_s"] = df_angles["frame"] / fps
        if "time_s" in df_angles.columns:
            df_angles = df_angles.sort_values("time_s")

    # Generate matplotlib charts
    temp_images = []
    
    def create_single_plot(df, x_col, y_col, title, ylabel, stats_func, filename):
        plt.figure(figsize=(7, 2.5), dpi=200)
        if df is not None and x_col in df.columns and y_col in df.columns:
            x = df[x_col].values
            y = df[y_col].values
            plt.plot(x, y, color="#1F77B4", linewidth=1.5)
            plt.fill_between(x, y, color="#1F77B4", alpha=0.1)
            plt.grid(True, linestyle="--", alpha=0.5)
            stats = stats_func(y)
        else:
            plt.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
            stats = "N/A"
        plt.title(title, fontsize=11, fontweight="bold", fontname="DejaVu Sans")
        plt.xlabel("Time (s)", fontsize=8, fontname="DejaVu Sans")
        plt.ylabel(ylabel, fontsize=8, fontname="DejaVu Sans")
        plt.tick_params(labelsize=8)
        plt.tight_layout()
        path = os.path.join(results_dir, filename)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        temp_images.append(path)
        return stats

    def create_paired_plot(df, x_col, left_col, right_col, title, ylabel, stats_func, filename):
        plt.figure(figsize=(7, 2.5), dpi=200)
        if df is not None and x_col in df.columns and left_col in df.columns and right_col in df.columns:
            x = df[x_col].values
            yl = df[left_col].values
            yr = df[right_col].values
            plt.plot(x, yl, color="#1F77B4", linewidth=1.5, label="Left")
            plt.plot(x, yr, color="#FF7F0E", linewidth=1.5, label="Right")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper right", fontsize=8)
            stats = stats_func(yl, yr)
        else:
            plt.text(0.5, 0.5, "Data Not Available", ha='center', va='center')
            stats = "N/A"
        plt.title(title, fontsize=11, fontweight="bold", fontname="DejaVu Sans")
        plt.xlabel("Time (s)", fontsize=8, fontname="DejaVu Sans")
        plt.ylabel(ylabel, fontsize=8, fontname="DejaVu Sans")
        plt.tick_params(labelsize=8)
        plt.tight_layout()
        path = os.path.join(results_dir, filename)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        temp_images.append(path)
        return stats

    # 1. Distance Chart
    def dist_stats(y):
        total = y[-1] if len(y) > 0 else 0.0
        return {"total": f"{total:.2f} m"}
    dist_text = create_single_plot(df_metrics, "time_s", "dist_m", "Distance", "Distance (m)", dist_stats, "temp_dist.png")

    # 2. Velocity Chart
    def vel_stats(y):
        if len(y) == 0:
            return "N/A"
        avg_v = np.mean(y)
        max_v = np.max(y)
        min_v = np.min(y)
        return {
            "avg": f"{avg_v:.3f} m/s",
            "max": f"{max_v:.3f} m/s",
            "min": f"{min_v:.3f} m/s"
        }
    vel_text = create_single_plot(df_metrics, "time_s", "speed_mps", "Velocity", "Velocity (m/s)", vel_stats, "temp_vel.png")

    # 3. Acceleration Chart
    def accel_stats(y):
        if len(y) == 0:
            return "N/A"
        avg_a = np.mean(y)
        max_a = np.max(y)
        min_a = np.min(y)
        return {
            "avg": f"{avg_a:.3f} m/s\u00b2",
            "max": f"{max_a:.3f} m/s\u00b2",
            "min": f"{min_a:.3f} m/s\u00b2"
        }
    accel_text = create_single_plot(df_metrics, "time_s", "accel_mps2", "Acceleration", "Acceleration (m/s\u00b2)", accel_stats, "temp_accel.png")

    # 4. Knee Angle Chart
    def knee_stats(yl, yr):
        if len(yl) == 0 or len(yr) == 0:
            return "N/A"
        return {
            "left_avg": f"{np.mean(yl):.1f}\u00b0", "left_max": f"{np.max(yl):.1f}\u00b0", "left_min": f"{np.min(yl):.1f}\u00b0",
            "right_avg": f"{np.mean(yr):.1f}\u00b0", "right_max": f"{np.max(yr):.1f}\u00b0", "right_min": f"{np.min(yr):.1f}\u00b0"
        }
    knee_text = create_paired_plot(df_angles, "time_s", "left_knee_angle", "right_knee_angle", "Knee Angle", "Angle (deg)", knee_stats, "temp_knee.png")

    # 5. Hip Angle Chart
    hip_text = create_paired_plot(df_angles, "time_s", "left_hip_angle", "right_hip_angle", "Hip Angle", "Angle (deg)", knee_stats, "temp_hip.png")

    # 6. Elbow Flexion Chart
    elbow_text = create_paired_plot(df_angles, "time_s", "left_elbow_flexion_angle", "right_elbow_flexion_angle", "Elbow Flexion", "Angle (deg)", knee_stats, "temp_elbow.png")

    # 7. Shoulder Flexion Chart
    shoulder_text = create_paired_plot(df_angles, "time_s", "left_shoulder_flexion", "right_shoulder_flexion", "Shoulder Flexion", "Angle (deg)", knee_stats, "temp_shoulder.png")

    # 8. Pelvis Torso Angle Chart
    def pelvis_stats(y):
        if len(y) == 0:
            return "N/A"
        avg_pt = np.mean(y)
        max_pt = np.max(y)
        min_pt = np.min(y)
        return {
            "avg": f"{avg_pt:.1f}\u00b0",
            "max": f"{max_pt:.1f}\u00b0",
            "min": f"{min_pt:.1f}\u00b0"
        }
    pelvis_text = create_single_plot(df_angles, "time_s", "pelvis_torso_angle", "Pelvis Torso Angle", "Angle (deg)", pelvis_stats, "temp_pelvis.png")

    # Setup FPDF PDF Document
    class CustomPDF(FPDF):
        def header(self):
            # Print page top header except on the first page
            if self.page_no() > 1:
                self.set_y(10)
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(150, 150, 150)
                self.cell(0, 5, "Runner Analysis Report", 0, 0, "L")
                self.set_font("Helvetica", "", 10)
                self.cell(0, 5, f"Session: {run_session_info.get('id')}", 0, 1, "R")
                self.line(15, 16, 195, 16)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Runner Analysis Report - Session: {run_session_info.get('id')}", 0, 0, "L")
            self.cell(0, 10, f"Page {self.page_no()}/3", 0, 0, "R")

    pdf = CustomPDF()
    pdf.set_auto_page_break(False, margin=15)
    pdf.set_margins(15, 15, 15)

    # ------------------ PAGE 1 ------------------
    pdf.add_page()
    
    # Title (y=15)
    pdf.set_y(15)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(51, 51, 51)
    pdf.cell(0, 10, "Runner Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(102, 102, 102)
    pdf.cell(0, 5, f"Session: {run_session_info.get('id')}", ln=True, align="C")
    pdf.ln(5)

    # Tables layout (Runner, Date, Status, etc.) (y=35)
    def draw_row(headers, values, widths):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(80, 80, 80)
        for i, h in enumerate(headers):
            pdf.cell(widths[i], 5, h, ln=0, align="L")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        for i, v in enumerate(values):
            pdf.cell(widths[i], 5, str(v), ln=0, align="L")
        pdf.ln(8)

    date_str = ""
    if run_session_info.get("date"):
        if isinstance(run_session_info["date"], datetime):
            date_str = run_session_info["date"].strftime('%Y-%m-%d %H:%M:%S')
        else:
            date_str = str(run_session_info["date"])

    draw_row(
        ["Runner", "Date", "Status"],
        [run_session_info.get("runnerName", "-"), date_str, run_session_info.get("status", "-")],
        [60, 80, 40]
    )
    
    draw_row(
        ["FPS", "Camera Count", "Total Time"],
        [f"{run_session_info.get('fps', '-')}", f"{run_session_info.get('cameraCount', '-')}", f"{run_session_info.get('totalTime', '-')} s" if run_session_info.get('totalTime') is not None else "-"],
        [60, 80, 40]
    )

    avg_vel = f"{run_session_info.get('avgVelocity'):.3f} m/s" if run_session_info.get('avgVelocity') is not None else "-"
    avg_acc = f"{run_session_info.get('avgAcceleration'):.3f} m/s\u00b2" if run_session_info.get('avgAcceleration') is not None else "-"
    avg_step = f"{run_session_info.get('avgStepLength'):.3f} m" if run_session_info.get('avgStepLength') is not None else "-"
    draw_row(
        ["Avg Velocity", "Avg Acceleration", "Avg Step Length"],
        [avg_vel, avg_acc, avg_step],
        [60, 80, 40]
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Note", ln=True, align="L")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 5, run_session_info.get("note") or "-", ln=True, align="L")
    
    # 1. Distance Chart (y=112)
    pdf.set_y(112)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(51, 51, 51)
    pdf.cell(0, 6, "Distance", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_dist.png"), x=25, y=118, w=160, h=57)
    pdf.set_y(176)
    if isinstance(dist_text, dict):
        draw_row(
            ["Total", "", "", "", "", ""],
            [dist_text.get("total", "-"), "", "", "", "", ""],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # 2. Velocity Chart (y=200)
    pdf.set_y(200)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Velocity", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_vel.png"), x=25, y=206, w=160, h=57)
    pdf.set_y(264)
    if isinstance(vel_text, dict):
        draw_row(
            ["Average", "Max", "Min", "", "", ""],
            [vel_text.get("avg", "-"), vel_text.get("max", "-"), vel_text.get("min", "-"), "", "", ""],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # ------------------ PAGE 2 ------------------
    pdf.add_page()
    
    # 3. Acceleration Chart (y=24)
    pdf.set_y(24)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Acceleration", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_accel.png"), x=25, y=30, w=160, h=57)
    pdf.set_y(88)
    if isinstance(accel_text, dict):
        draw_row(
            ["Average", "Max", "Min", "", "", ""],
            [accel_text.get("avg", "-"), accel_text.get("max", "-"), accel_text.get("min", "-"), "", "", ""],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # 4. Knee Angle Chart (y=112)
    pdf.set_y(112)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Knee Angle", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_knee.png"), x=25, y=118, w=160, h=57)
    pdf.set_y(176)
    if isinstance(knee_text, dict):
        draw_row(
            ["Left Avg", "Left Max", "Left Min", "Right Avg", "Right Max", "Right Min"],
            [knee_text['left_avg'], knee_text['left_max'], knee_text['left_min'], knee_text['right_avg'], knee_text['right_max'], knee_text['right_min']],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # 5. Hip Angle Chart (y=200)
    pdf.set_y(200)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Hip Angle", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_hip.png"), x=25, y=206, w=160, h=57)
    pdf.set_y(264)
    if isinstance(hip_text, dict):
        draw_row(
            ["Left Avg", "Left Max", "Left Min", "Right Avg", "Right Max", "Right Min"],
            [hip_text['left_avg'], hip_text['left_max'], hip_text['left_min'], hip_text['right_avg'], hip_text['right_max'], hip_text['right_min']],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # ------------------ PAGE 3 ------------------
    pdf.add_page()

    # 6. Elbow Flexion Chart (y=24)
    pdf.set_y(24)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Elbow Flexion", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_elbow.png"), x=25, y=30, w=160, h=57)
    pdf.set_y(88)
    if isinstance(elbow_text, dict):
        draw_row(
            ["Left Avg", "Left Max", "Left Min", "Right Avg", "Right Max", "Right Min"],
            [elbow_text['left_avg'], elbow_text['left_max'], elbow_text['left_min'], elbow_text['right_avg'], elbow_text['right_max'], elbow_text['right_min']],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # 7. Shoulder Flexion Chart (y=112)
    pdf.set_y(112)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Shoulder Flexion", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_shoulder.png"), x=25, y=118, w=160, h=57)
    pdf.set_y(176)
    if isinstance(shoulder_text, dict):
        draw_row(
            ["Left Avg", "Left Max", "Left Min", "Right Avg", "Right Max", "Right Min"],
            [shoulder_text['left_avg'], shoulder_text['left_max'], shoulder_text['left_min'], shoulder_text['right_avg'], shoulder_text['right_max'], shoulder_text['right_min']],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # 8. Pelvis Torso Angle Chart (y=200)
    pdf.set_y(200)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 6, "Pelvis Torso Angle", ln=True, align="L")
    pdf.image(os.path.join(results_dir, "temp_pelvis.png"), x=25, y=206, w=160, h=57)
    pdf.set_y(264)
    if isinstance(pelvis_text, dict):
        draw_row(
            ["Average", "Max", "Min", "", "", ""],
            [pelvis_text.get("avg", "-"), pelvis_text.get("max", "-"), pelvis_text.get("min", "-"), "", "", ""],
            [30, 30, 30, 30, 30, 30]
        )
    else:
        pdf.cell(0, 4, "N/A", ln=True)

    # Output PDF
    pdf.output(output_pdf_path)

    # Cleanup temp images
    for img_path in temp_images:
        try:
            os.remove(img_path)
        except Exception:
            pass
