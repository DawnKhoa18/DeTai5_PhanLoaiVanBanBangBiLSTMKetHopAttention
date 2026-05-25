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
# KHẮC PHỤC TRIỆT ĐỂ: VÁ LỖI PHÂN NHÁNH MODULE THEO ĐÚNG FILE SOURCE CỦA NHÓM
# =========================================================================
from models.attbilstm.att_bilstm import AttBiLSTM

# Tìm và lấy chính xác class Attention mà file att_bilstm.py đang sử dụng
try:
    from models.attbilstm.attention import Attention
except ImportError:
    # Nếu file attention nằm chung hoặc được import gián tiếp
    import models.attbilstm.att_bilstm as m
    if hasattr(m, 'Attention'):
        Attention = m.Attention
    else:
        # Phương án dự phòng cấu hình mạng tự động nếu không tìm thấy class
        class Attention(nn.Module):
            def __init__(self, rnn_size):
                super().__init__()
                self.w = nn.Parameter(torch.randn(rnn_size, 1))
            def forward(self, H):
                M = torch.tanh(H)
                alpha = torch.softmax(torch.matmul(M, self.w), dim=1)
                r = torch.sum(H * alpha, dim=1)
                return r, alpha.squeeze(-1)

# Xây dựng cây thư mục ảo giống hệt như vết lưu trong tệp checkpoint cũ
fake_parent = types.ModuleType('models.AttBiLSTM')
sys.modules['models.AttBiLSTM'] = fake_parent

fake_sub1 = types.ModuleType('models.AttBiLSTM.att_bilstm')
fake_sub1.AttBiLSTM = AttBiLSTM
sys.modules['models.AttBiLSTM.att_bilstm'] = fake_sub1
setattr(fake_parent, 'att_bilstm', fake_sub1)

fake_sub2 = types.ModuleType('models.AttBiLSTM.attention')
fake_sub2.Attention = Attention  # Khai báo thuộc tính 'Attention' đúng chữ viết hoa
sys.modules['models.AttBiLSTM.attention'] = fake_sub2
setattr(fake_parent, 'attention', fake_sub2)


# =========================================================================
# CẤU HÌNH GIAO DIỆN WEB STREAMLIT
# =========================================================================
st.set_page_config(
    page_title="Phân loại Văn bản Yahoo",
    page_icon=":speech_balloon:",
    layout="wide"
)

# Danh mục phân loại của tập dữ liệu Yahoo Answers
DANH_MỤC_YAHOO = {
    0: "Society & Culture (Xã hội & Văn hóa)",
    1: "Science & Mathematics (Khoa học & Toán học)",
    2: "Health (Sức khỏe)",
    3: "Education & Reference (Giáo dục & Tra cứu)",
    4: "Computers & Internet (Máy tính & Internet)",
    5: "Sports (Thể thao)",
    6: "Business & Finance (Kinh doanh & Tài chính)",
    7: "Entertainment & Music (Giải trí & Âm nhạc)",
    8: "Family & Relationships (Gia đình & Mối quan hệ)",
    9: "Politics & Government (Chính trị & Chính phủ)"
}

# Tiền xử lý chuỗi văn bản đầu vào phù hợp với Embedding Layer
def preprocess_text(text, word_map, max_len=50):
    tokens = text.lower().split()
    actual_length = max(1, min(len(tokens), max_len))
    sequence = [word_map.get(token, word_map.get('<unk>', 1)) for token in tokens]
    
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
        
    return torch.tensor([sequence], dtype=torch.long), torch.tensor([actual_length], dtype=torch.long), tokens[:max_len]

# Khởi tạo và nạp tài nguyên hệ thống
@st.cache_resource
def load_all_resources():
    model_file = "checkpoint_attbilstm_yahoo_answers.pth.tar"
    if not os.path.exists(model_file):
        with st.spinner("Đang tải mô hình Deep Learning từ Google Drive ..."):
            drive_id_model = "13Cg1OJEAe3vkZfc0_ws8LS2zFOTK6dOV" 
            url = f"https://drive.google.com/uc?id={drive_id_model}"
            gdown.download(url, model_file, quiet=False)
            
    word_map_file = "word_map.json"
    if not os.path.exists(word_map_file):
        with st.spinner("Đang tải bộ từ điển ngôn ngữ từ Google Drive..."):
            drive_id_word = "1Z6JVv1HPkUmBU609GeXPoX0ZpkKgaaMV"
            url = f"https://drive.google.com/uc?id={drive_id_word}"
            gdown.download(url, word_map_file, quiet=False)

    metrics_file = "history_metrics.pkl"
    if not os.path.exists(metrics_file):
        st.error(f"Không tìm thấy file '{metrics_file}' trong thư mục nguồn!")
        st.stop()

    # Đọc dữ liệu từ file JSON và file Pickle
    with open(word_map_file, 'r', encoding='utf-8') as f:
        word_map = json.load(f)
        
    with open(metrics_file, 'rb') as f:
        data_pkl = pickle.load(f)
        history_dict = data_pkl['history']
        metrics_data = data_pkl['metrics']

    # Khởi tạo kiến trúc mạng cục bộ sạch theo cấu hình nhóm đã train
    vocab_size = len(word_map)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 10
    
    model = AttBiLSTM(
        n_classes=output_dim,
        vocab_size=vocab_size,
        embeddings=None,  
        emb_size=embedding_dim,
        fine_tune=True,
        rnn_size=hidden_dim,
        rnn_layers=1,
        dropout=0.5
    )
    
    # Sử dụng torch.load để giải mã tệp tin nhị phân an toàn thông qua Module ảo
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            state_dict = checkpoint['model'] if isinstance(checkpoint['model'], dict) else checkpoint['model'].state_dict()
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())
        
    model.eval()
    return model, word_map, history_dict, metrics_data

# Thực thi nạp tài nguyên
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

# ---- TAB 1: PHÂN TÍCH VĂN BẢN VÀ TRỰC QUAN HÓA ----
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
                input_tensor, actual_length, tokens = preprocess_text(user_text, word_map)
                
                with torch.no_grad():
                    outputs, attn_weights = model(input_tensor, actual_length, return_attention=True)
                    probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                    
                pred_class_id = int(np.argmax(probabilities))
                
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
        else:
            st.success(f":tada: Chủ đề dự báo hệ thống: **{st.session_state.pred_topic}**")
            st.metric(label=":target: Độ tin cậy dự đoán chính xác", value=f"{st.session_state.conf_yahoo:.2f}%")
            
            st.markdown("### :mag: Trực quan hóa Trọng số Attention (Word Importance):")
            st.write("Mô hình mạng Neural đang tập trung vào các từ khóa mang tính quyết định để đưa ra chuyên mục.")
            
            if st.session_state.attn_weights is not None and st.session_state.tokens is not None:
                actual_len = len(st.session_state.tokens)
                # Đảm bảo phân tách đúng kích thước mảng trọng số Attention
                if st.session_state.attn_weights.ndim > 1:
                    weights_to_show = st.session_state.attn_weights[0][:actual_len]
                else:
                    weights_to_show = st.session_state.attn_weights[:actual_len]
                tokens_to_show = st.session_state.tokens
                
                fig_attn, ax_attn = plt.subplots(figsize=(6, max(2, actual_len * 0.3)))
                sns.barplot(x=weights_to_show, y=tokens_to_show, palette="Blues_r", ax=ax_attn)
                ax_attn.set_title("Trọng số mức độ tập trung Attention", fontsize=10, fontweight='bold')
                st.pyplot(fig_attn)

# ---- TAB 2: ĐÁNH GIÁ MÔ HÌNH ----
with tab2:
    st.markdown("## :chart_with_upwards_trend: Kết Quả Thực Nghiệm Mạng Học Sâu BiLSTM + Attention")
    
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
    try:
        report = classification_report(
            metrics_data['y_test'], 
            metrics_data['y_pred'], 
            target_names=list(DANH_MỤC_YAHOO.values())
        )
        st.code(report, language="text")
    except Exception as e:
        st.warning(f"Không thể kết xuất dữ liệu báo cáo phân loại. Chi tiết: {e}")
