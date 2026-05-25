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
import sys
import types

# =========================================================================
# GỌI KIẾN TRÚC MẠNG CHUẨN XÁC TỪ THƯ MỤC VỪA TẠO TRÊN GITHUB
# Không cần sử dụng các đoạn mã tạo module giả lập (sys.modules['models'])
# =========================================================================
from models.attbilstm.att_bilstm import AttBiLSTM

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

# Hàm tiền xử lý chuỗi văn bản đầu vào khớp với cấu hình huấn luyện
def preprocess_text(text, word_map, max_len=50):
    # Loại bỏ ký tự đặc biệt cơ bản và chuyển về chữ thường
    tokens = text.lower().split()
    
    # Chuyển từ sang index dựa trên word_map (mặc định lấy 1 nếu là từ không có trong từ điển <unk>)
    sequence = [word_map.get(token, word_map.get('<unk>', 1)) for token in tokens]
    
    # Cắt ngắn hoặc đệm thêm số 0 (<pad>) cho đủ độ dài max_len
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
        
    return torch.tensor([sequence], dtype=torch.long), tokens[:max_len]

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

    # Khởi tạo cấu hình tham số mạng Neural thực tế
    vocab_size = len(word_map)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 10
    
    model = AttBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # Nạp trọng số từ checkpoint (Ép chạy trên CPU)
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
        model.load_state_dict(checkpoint['model'].state_dict())
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
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
            st.session_state.attn_weights = None
            st.session_state.tokens = None

        if st.button(":rocket: Tiến hành phân tích chủ đề", type="primary", width="stretch"):
            if user_text.strip() == "":
                st.warning("Vui lòng nhập văn bản trước khi bấm nút dự đoán!")
            else:
                # 1. Tiền xử lý chuỗi nhập vào thực tế
                input_tensor, tokens = preprocess_text(user_text, word_map)
                
                # 2. Đưa dữ liệu qua mạng Neural dự đoán thực tế
                with torch.no_grad():
                    outputs, attn_weights = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                    
                # 3. Tính toán nhãn có xác suất cao nhất từ kết quả mạng Neural
                pred_class_id = int(np.argmax(probabilities))
                
                # Cập nhật kết quả tính toán thật lên giao diện ứng dụng
                st.session_state.pred_topic = DANH_MỤC_YAHOO[pred_class_id]
                st.session_state.conf_yahoo = probabilities[pred_class_id] * 100
                st.session_state.prob_yahoo = probabilities
                st.session_state.attn_weights = attn_weights.squeeze().numpy()
                st.session_state.tokens = tokens

        if st.session_state.prob_yahoo is not None:
            st.markdown("### :bar_chart: Phân phối xác suất các chuyên mục:")
            proba_df = pd.DataFrame({
                'Chuyên mục': list(DANH_MỤC_YAHOO.values()),
                'Xác suất (%)': st.session_state.prob_yahoo * 100
            })
            st.bar_chart(data=proba_df, x='Chuyên mục', y='Xác suất (%)', width="stretch")

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
            
            # Vẽ biểu đồ ngang thể hiện mức độ quan trọng (Attention Weights) của từng từ
            if st.session_state.attn_weights is not None and st.session_state.tokens is not None:
                actual_len = len(st.session_state.tokens)
                weights_to_show = st.session_state.attn_weights[:actual_len]
                tokens_to_show = st.session_state.tokens
                
                fig_attn, ax_attn = plt.subplots(figsize=(6, max(2, actual_len * 0.3)))
                sns.barplot(x=weights_to_show, y=tokens_to_show, palette="Blues_r", ax=ax_attn)
                ax_attn.set_title("Trọng số mức độ tập trung Attention", fontsize=10, fontweight='bold')
                st.pyplot(fig_attn)
                
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
        report = classification_report(
            metrics_data['y_test'], 
            metrics_data['y_pred'], 
            target_names=list(DANH_MỤC_YAHOO.values())
        )
        st.code(report, language="text")
    except Exception as e:
        st.warning(f"Không thể kết xuất dữ liệu báo cáo phân loại. Chi tiết: {e}")
        
    st.info(":light_bulb: **Chú thích ý nghĩa các chỉ số:**\n"
            "- **Precision (Độ chính xác dự báo):** Trong số các mẫu được hệ thống xếp vào chủ đề này, có bao nhiêu phần trăm là đúng thực tế.\n"
            "- **Recall (Độ phủ/Tỉ lệ tìm sót):** Trong số tất cả các mẫu của chủ đề này có trong tập kiểm thử, hệ thống đã nhận diện được bao nhiêu phần trăm.\n"
            "- **F1-score:** Chỉ số đánh giá cân bằng (trung bình điều hòa) giữa cả hai yếu tố trên nhằm phản ánh hiệu năng tổng quát.")
