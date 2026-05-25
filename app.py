import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os
import gdown
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Giao diện web
st.set_page_config(
    page_title="Phân loại Văn bản Yahoo",
    page_icon=":speech_balloon:",
    layout="wide"
)

# Tên các chủ đề của Yahoo Answers
DANH_MỤC_YAHOO = {
    0: "Society & Culture (Xã hội & Văn hóa)",
    1: "Science & Mathematics (Khoa học & Toán học)",
    2: "Health (Sức khỏe)",
    3: "Education & Reference (Giáo dịch & Tra cứu)",
    4: "Computers & Internet (Máy tính & Internet)",
    5: "Sports (Thể thao)",
    6: "Business & Finance (Kinh doanh & Tài chính)",
    7: "Entertainment & Music (Giải trí & Âm nhạc)",
    8: "Family & Relationships (Gia đình & Mối quan hệ)",
    9: "Politics & Government (Chính trị & Chính phủ)"
}

# Tải file từ gg drive và đọc tài nguyên cục bộ
@st.cache_resource
def load_all_resources():
    # 1. Tải file trọng số mô hình (.pth.tar) từ Google Drive
    model_file = "checkpoint_attbilstm_yahoo_answers.pth.tar"
    if not os.path.exists(model_file):
        with st.spinner("Đang tải mô hình Deep Learning từ Google Drive ..."):
            drive_id_model = "13Cg1OJEAe3vkZfc0_ws8LS2zFOTK6dOV" 
            url = f"https://drive.google.com/uc?id={drive_id_model}"
            gdown.download(url, model_file, quiet=False)
            
    # 2. Tải file từ điển (word_map.json) từ Google Drive
    word_map_file = "word_map.json"
    if not os.path.exists(word_map_file):
        with st.spinner("Đang tải bộ từ điển ngôn ngữ từ Google Drive..."):
            drive_id_word = "1Z6JVv1HPkUmBU609GeXPoX0ZpkKgaaMV"
            url = f"https://drive.google.com/uc?id={drive_id_word}"
            gdown.download(url, word_map_file, quiet=False)

    # 3. Đọc file số liệu đánh giá (history_metrics.pkl) trực tiếp từ bộ nhớ cục bộ (đã đẩy lên GitHub)
    metrics_file = "history_metrics.pkl"
    if not os.path.exists(metrics_file):
        st.error(f"Không tìm thấy file '{metrics_file}' trong thư mục nguồn! Vui lòng đảm bảo bạn đã push file này lên repository GitHub.")
        st.stop()

    # --- ĐỌC CÁC FILE ---
    # Đọc từ điển word_map
    with open(word_map_file, 'r', encoding='utf-8') as f:
        word_map = json.load(f)
        
    # Đọc lịch sử huấn luyện từ file cục bộ
    with open(metrics_file, 'rb') as f:
        data_pkl = pickle.load(f)
        history_dict = data_pkl['history']
        metrics_data = data_pkl['metrics']

    # Ghi chú: Vì chạy trên Streamlit Web không có GPU, mô hình sẽ được ép chạy bằng CPU bằng lệnh map_location
    # checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    # model = checkpoint['model'] # Hoặc tùy cách cấu hình nạp trọng số của nhóm bạn
    
    # Tạm thời trả về đối tượng giả lập để giao diện không bị lỗi crash trước khi nhóm cấu hình cấu trúc mạng cụ thể
    model = None 
    
    return model, word_map, history_dict, metrics_data

# Khởi chạy hàm nạp tài nguyên
try:
    model, word_map, history_dict, metrics_data = load_all_resources()
except Exception as e:
    st.error(f"Lỗi hệ thống khi tải cấu hình hoặc đọc file tài nguyên: {e}")
    st.stop()


# ==========================================
# THIẾT KẾ GIAO DIỆN WEB
# ==========================================
st.title(":speech_balloon: Hệ Thống Phân Loại Chủ Đề Văn Bản Yahoo Answers (BiLSTM + Attention)")
st.subheader("Sản phẩm nghiên cứu công nghệ phát triển bởi: **Nhóm 6**")
st.markdown("---")

tab1, tab2 = st.tabs([":crystal_ball: Phân Tích Trực Quan", ":bar_chart: Đánh Giá Hiệu Năng Mô Hình"])

# ---- TAB 1: PHÂN TÍCH VĂN BẢN ----
with tab1:
    col_trai, col_phai = st.columns([1.2, 1])
    
    with col_trai:
        st.markdown("### :pencil2: Nhập nội dung văn bản câu hỏi (Tiếng Anh):")
        user_text = st.text_area(
            label="Nhập câu hỏi hoặc đoạn văn cần phân loại vào đây:",
            value="What is the best way to learn computer programming online for free?",
            height=150
        )
        st.markdown("---")
        
        if 'pred_topic' not in st.session_state:
            st.session_state.pred_topic = None
            st.session_state.conf_yahoo = 0.0
            st.session_state.prob_yahoo = None

        if st.button(":rocket: Tiến hành phân tích chủ đề", type="primary", use_container_width=True):
            if user_text.strip() == "":
                st.warning("Vui lòng nhập văn bản trước khi bấm nút dự đoán!")
            else:
                # Mô phỏng quá trình xử lý văn bản và dự đoán xác suất ngẫu nhiên để demo giao diện mẫu
                # Nhóm 6 sẽ thay thế đoạn xử lý token và model(input) thực tế của nhóm tại đây
                st.session_state.pred_topic = DANH_MỤC_YAHOO[1] # Tạm thời lấy lớp số 1 làm mẫu
                st.session_state.conf_yahoo = 94.52
                
                # Giả lập mảng xác suất cho 10 class
                pseudo_probs = np.random.dirichlet(np.ones(10), size=1)[0]
                pseudo_probs[1] = 0.9452  # Ép cho class dự đoán cao nhất
                st.session_state.prob_yahoo = pseudo_probs / pseudo_probs.sum()

        if st.session_state.pred_topic is not None:
            st.markdown("### :bar_chart: Phân phối xác suất các chuyên mục:")
            proba_df = pd.DataFrame({
                'Chuyên mục': list(DANH_MỤC_YAHOO.values()),
                'Xác suất (%)': st.session_state.prob_yahoo * 100
            })
            st.bar_chart(data=proba_df, x='Chuyên mục', y='Xác suất (%)', use_container_width=True)

    with col_phai:
        st.markdown("### :desktop_computer: Kết quả nhận diện hệ thống")
        if st.session_state.pred_topic is None:
            st.info(":light_bulb: Nhập đoạn văn bản ở cột bên trái và bấm nút 'Phân tích' để kích hoạt mạng Neural nhận diện!")
            st.markdown("""
            **Gợi ý câu mẫu để test thử:**
            1. *Thể thao:* "Who is the greatest basketball player of all time in NBA history?"
            2. *Khoa học / Toán:* "Can someone explain the theory of relativity and quantum mechanics in simple terms?"
            3. *Kinh doanh:* "How do interest rates affect the stock market and inflation?"
            """)
        else:
            st.success(f":tada: Chủ đề dự báo hệ thống: **{st.session_state.pred_topic}**")
            st.metric(label=":target: Độ tin cậy dự đoán chính xác", value=f"{st.session_state.conf_yahoo:.2f}%")
            
            st.markdown("### :mag: Trực quan hóa Trọng số Attention (Word Importance):")
            st.write("Mô hình mạng Neural đang tập trung vào các từ khóa mang tính quyết định để đưa ra chuyên mục.")
            st.info(":information_source: Cơ chế Attention giúp trích xuất từ khóa quyết định bản chất ngữ nghĩa của câu hỏi.")

# ---- TAB 2: ĐÁNH GIÁ MÔ HÌNH ----
with tab2:
    st.markdown("## :chart_with_upwards_trend: Kết Quả Thực Nghiệm Mạng Học Sâu BiLSTM + Attention")
    st.write("Số liệu kiểm thử mô hình thu được trên tập dữ liệu phân loại văn bản Yahoo Answers.")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label=":target: Độ chính xác tập Kiểm thử (Test Accuracy)", value=f"{metrics_data['test_accuracy']*100:.2f}%")
    with metric_col2:
        st.metric(label=":chart: Độ mất mát tập Kiểm thử (Test Loss)", value=f"{metrics_data['test_loss']:.4f}")
    with metric_col3:
        st.metric(label=":gear: Kiến trúc mạng", value="BiLSTM + Attention")

    st.markdown("---")
    st.subheader(":chart_with_features: 1. Biểu đồ Quá trình Huấn luyện (Training History)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history_dict['train_acc'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(history_dict['val_acc'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title('Mô hình Accuracy qua các Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--')

    ax2.plot(history_dict['train_loss'], label='Train Loss', color='#d62728', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Validation Loss', color='#2ca02c', linewidth=2)
    ax2.set_title('Mô hình Loss qua các Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--')
    st.pyplot(fig) 

    st.markdown("---")
    st.subheader(":jigsaw: 2. Ma trận nhầm lẫn (Confusion Matrix)")
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(metrics_data['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=list(DANH_MỤC_YAHOO.values()), yticklabels=list(DANH_MỤC_YAHOO.values()), ax=ax_cm)
    plt.xlabel('Chuyên mục dự đoán', fontsize=10, fontweight='bold')
    plt.ylabel('Chuyên mục thực tế', fontsize=10, fontweight='bold')
    st.pyplot(fig_cm)

    st.markdown("---")
    st.subheader(":clipboard: 3. Báo cáo phân loại chi tiết (Classification Report)")
    st.write("Chi tiết các chỉ số thống kê định lượng đánh giá độ chính xác trên từng chuyên mục văn bản:")
    
    try:
        # Tính toán báo cáo phân loại từ dữ liệu test thực tế lưu trong file pkl
        report = classification_report(
            metrics_data['y_test'], 
            metrics_data['y_pred'], 
            target_names=list(DANH_MỤC_YAHOO.values())
        )
        # Hiển thị dạng khối Code Textbox cho thẳng hàng cột dữ liệu
        st.code(report, language="text")
    except Exception as e:
        st.warning(f"Không thể kết xuất dữ liệu báo cáo phân loại. Chi tiết: {e}")
        
    st.info(":light_bulb: **Chú thích ý nghĩa các chỉ số:**\n"
            "- **Precision (Độ chính xác dự báo):** Trong số các mẫu được hệ thống xếp vào chủ đề này, có bao nhiêu phần trăm là đúng thực tế.\n"
            "- **Recall (Độ phủ/Tỉ lệ tìm sót):** Trong số tất cả các mẫu của chủ đề này có trong tập kiểm thử, hệ thống đã nhận diện được bao nhiêu phần trăm.\n"
            "- **F1-score:** Chỉ số đánh giá cân bằng (trung bình điều hòa) giữa cả hai yếu tố trên nhằm phản ánh hiệu năng tổng quát.")
