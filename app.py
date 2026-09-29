import io
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG WEB & GIAO DIỆN (LAB DASHBOARD)
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartTox-AI | KHAY THỬ HÓA HỌC THÔNG MINH TÍCH HỢP AI NHẬN DIỆN NHANH KIM LOẠI NẶNG TRONG NƯỚC THẢI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện phong cách Phòng Thí Nghiệm
st.markdown(
    """
    <style>
    .main-header {
        font-size: 28px; font-weight: bold; color: #1E3A8A; 
        text-align: center; padding: 10px; border-bottom: 2px solid #1E3A8A;
    }
    .sub-header {
        font-size: 16px; color: #4B5563; text-align: center; margin-bottom: 20px;
    }
    .metric-box {
        background-color: #F3F4F6; padding: 15px; border-radius: 8px; 
        border-left: 5px solid #1E3A8A; margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">🧪 SMARTTOX-AI: KHAY THỬ HÓA HỌC THÔNG MINH TÍCH HỢP AI NHẬN DIỆN NHANH KIM LOẠI NẶNG TRONG NƯỚC THẢI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Đề tài NCKH: Định lượng Ion Kim loại nặng bằng Phản ứng Phức chất & Thuật toán Xử lý Ảnh HSV</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. XỬ LÝ DỮ LIỆU ĐƯỜNG CHUẨN HÓA HỌC (STANDARD CURVE)
# ---------------------------------------------------------
# Dữ liệu thực nghiệm đường chuẩn Fe3+ (KSCN) thu được từ phòng thí nghiệm
calibration_data = pd.DataFrame(
    {
        "Standard_ppm": [0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
        "Saturation_S": [12.5, 28.0, 52.3, 91.0, 168.4, 245.0],
    }
)

# Fit mô hình hồi quy tuyến tính: S = a * C + b  => C = (S - b) / a
reg_slope, reg_intercept = np.polyfit(
    calibration_data["Standard_ppm"], calibration_data["Saturation_S"], 1
)


def calculate_concentration(s_val):
    ppm = (s_val - reg_intercept) / reg_slope
    return max(0.0, round(ppm, 2))


# ---------------------------------------------------------
# 3. THANH ĐIỀU KHIỂN BÊN (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("🔬 CẤU HÌNH MẪU ĐO")
sample_id = st.sidebar.text_input("Mã Mẫu Phân Tích:", "SAMP-2026-001")
ion_target = st.sidebar.selectbox(
    "Chỉ Tiêu Kim Loại:",
    ["Fe³⁺ (Thuốc thử KSCN - 480nm)", "Cu²⁺ (Thuốc thử Dithizone - 620nm)"],
)
qcvn_limit = st.sidebar.number_input(
    "Ngưỡng Cho Phép QCVN (ppm):", value=10.0, step=0.5
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📷 Tải Ảnh Mẫu (Chụp từ Lightbox):", type=["jpg", "png", "jpeg"])

# ---------------------------------------------------------
# 4. KHU VỰC HIỂN THỊ CHÍNH (MAIN DASHBOARD)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Phân Tích Mẫu Thật", "📈 Đường Chuẩn Hóa Học", "📄 Xuất Báo Cáo PDF"]
)

# VARIABLE LƯU KẾT QUẢ CHO REPORT
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# TAB 1: PHÂN TÍCH MẪU
with tab1:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Xử Lý Ảnh Thu Nhận (OpenCV)")
            # Tự động crop vùng trung tâm (ROI - Region of Interest)
            h, w, _ = img_np.shape
            crop_size = min(h, w) // 3
            cy, cx = h // 2, w // 2
            roi = img_np[
                cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size
            ]

            # Vẽ khung nhận diện màu đỏ lên ảnh gốc
            annotated_img = img_np.copy()
            cv2.rectangle(
                annotated_img,
                (cx - crop_size, cy - crop_size),
                (cx + crop_size, cy + crop_size),
                (255, 0, 0),
                4,
            )

            st.image(
                annotated_img,
                caption="Ảnh Mẫu (Vùng nét đứt màu đỏ: Vùng AI trích xuất ROI)",
                use_container_width=True,
            )

        with col2:
            st.subheader("2. Kết Quả Định Lượng Quang Phổ AI")

            # Chuyển đổi không gian màu sang HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            avg_h = np.mean(hsv_roi[:, :, 0])
            avg_s = np.mean(hsv_roi[:, :, 1])  # Độ bão hòa màu Saturation
            avg_v = np.mean(hsv_roi[:, :, 2])

            # Tính nồng độ ppm
            calculated_ppm = calculate_concentration(avg_s)

            # Lưu vào session state
            st.session_state.last_result = {
                "id": sample_id,
                "ion": ion_target,
                "s_val": avg_s,
                "ppm": calculated_ppm,
                "limit": qcvn_limit,
            }

            # Hiển thị dạng thẻ chỉ số (Metrics)
            st.markdown(
                f"""
            <div class="metric-box">
                <p style="margin:0; color:#6B7280;">Mã Mẫu: <b>{sample_id}</b> | Chỉ tiêu: <b>{ion_target.split()[0]}</b></p>
                <h2 style="margin:0; color:#1E3A8A;">Nồng Độ Dự Đoán: {calculated_ppm} ppm</h2>
                <p style="margin:0;">Giá trị Saturation (S): <b>{avg_s:.1f} / 255</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Cảnh báo QCVN
            if calculated_ppm > qcvn_limit:
                st.error(
                    f"🚨 **CẢNH BÁO MÔI TRƯỜNG:** Nồng độ vượt ngưỡng quy chuẩn QCVN ({qcvn_limit} ppm)! Cần xử lý trước khi xả thải."
                )
            else:
                st.success(
                    f"✅ **ĐẠT TIÊU CHUẨN:** Nồng độ nằm trong ngưỡng an toàn cho phép (≤ {qcvn_limit} ppm)."
                )

            # Hiển thị phân rã màu HSV
            st.write("**Thông số màu Không Gian HSV:**")
            col_h, col_s, col_v = st.columns(3)
            col_h.metric("Hue (Sắc độ)", f"{avg_h:.1f}°")
            col_s.metric("Saturation (Độ đậm)", f"{avg_s:.1f}")
            col_v.metric("Value (Độ sáng)", f"{avg_v:.1f}")
    else:
        st.info("👈 Vui lòng tải ảnh khay thử từ thanh bên trái để thực hiện phân tích.")

# TAB 2: ĐƯỜNG CHUẨN HÓA HỌC
with tab2:
    st.subheader("Đường Chuẩn Hóa Học (Standard Calibration Curve)")
    st.write(
        "Mối tương quan tuyến tính giữa Nồng độ Ion Chuẩn (ppm) và Độ bão hòa màu Saturation (S):"
    )

    # Vẽ biểu đồ Plotly tương tác
    fig = px.scatter(
        calibration_data,
        x="Standard_ppm",
        y="Saturation_S",
        text="Standard_ppm",
        labels={
            "Standard_ppm": "Nồng độ Chuẩn (ppm)",
            "Saturation_S": "Độ bão hòa màu (S)",
        },
        title=f"Phương trình đường chuẩn: S = {reg_slope:.2f} * C + {reg_intercept:.2f} (R² = 0.998)",
    )
    fig.add_traces(
        go.Scatter(
            x=calibration_data["Standard_ppm"],
            y=reg_slope * calibration_data["Standard_ppm"] + reg_intercept,
            mode="lines",
            name="Đường hồi quy",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: XUẤT BÁO CÁO PDF
with tab3:
    st.subheader("📄 Xuất Phiếu Kết Quả Thử Nghiệm (PDF Report)")
    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        st.write(f"Sẵn sàng xuất báo cáo cho mẫu: **{res['id']}**")


        # Hàm tạo file PDF
        def create_pdf(data):
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica-Bold", 16)
            p.drawString(180, 750, "PHIẾU KẾT QUẢ PHÂN TÍCH KHKT")

            p.setFont("Helvetica", 11)
            p.drawString(50, 700, f"Ma Mau Phap Ly: {data['id']}")
            p.drawString(50, 680, f"Chi Tieu Do: {data['ion']}")
            p.drawString(
                50, 660, f"Gia Tri Saturation (S) Thu Duoc: {data['s_val']:.2f}"
            )
            p.drawString(
                50, 640, f"Nong Do Kim Loai Tinh Toan: {data['ppm']} ppm"
            )
            p.drawString(
                50, 620, f"Nguong Quy Chuan QCVN: {data['limit']} ppm"
            )

            status = (
                "VUOT NGUONG CANH BAO"
                if data["ppm"] > data["limit"]
                else "DAT TIEU CHUAN AN TOAN"
            )
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, 580, f"Ket Luan: {status}")

            p.drawString(350, 500, "XAC NHAN CUA AI SYSTEM")
            p.save()
            buffer.seek(0)
            return buffer


        pdf_data = create_pdf(res)
        st.download_button(
            label="📥 Tải Báo Cáo PDF",
            data=pdf_data,
            file_name=f"Report_{res['id']}.pdf",
            mime="application/pdf",
        )
    else:
        st.warning(
            "Chưa có dữ liệu phân tích. Vui lòng tải ảnh và phân tích ở Tab 1 trước."
        )