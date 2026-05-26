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

from models.attbilstm.att_bilstm import AttBiLSTM
from models.attbilstm.attention import Attention

fake_parent = types.ModuleType('models.AttBiLSTM')
sys.modules['models.AttBiLSTM'] = fake_parent

fake_sub1 = types.ModuleType('models.AttBiLSTM.att_bilstm')
fake_sub1.AttBiLSTM = AttBiLSTM
sys.modules['models.AttBiLSTM.att_bilstm'] = fake_sub1
setattr(fake_parent, 'att_bilstm', fake_sub1)

fake_sub2 = types.ModuleType('models.AttBiLSTM.attention')
fake_sub2.Attention = Attention
sys.modules['models.AttBiLSTM.attention'] = fake_sub2
setattr(fake_parent, 'attention', fake_sub2)

# Cấu trúc giao diện 
st.set_page_config(
    page_title="Phân loại Văn bản Yahoo Answers",
    page_icon=":speech_balloon:",
    layout="wide"
)

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

# Hàm tách từ tương thích cao với word_map.json của hệ thống câu hỏi Yahoo
def preprocess_text(text, word_map, max_len=50):
    # Chuẩn hóa khoảng trống cho các ký tự đặc biệt
    for punct in ['.', '?', '!', ',', '(', ')', ':', ';']:
        text = text.replace(punct, f" {punct} ")
        
    tokens = text.lower().split()
    if len(tokens) == 0:
        tokens = ["<unk>"]
        
    actual_length = max(1, min(len(tokens), max_len))
    
    # Ánh xạ từ sang ID, dùng ID 1 nếu từ đó không nằm trong bộ từ điển
    sequence = [word_map.get(token, 1) for token in tokens]
    
    # Thực hiện Padding cho đủ độ dài
    if len(sequence) < max_len:
        tokens_padded = tokens + ["<pad>"] * (max_len - len(sequence))
        sequence = sequence + [0] * (max_len - len(sequence))
    else:
        tokens_padded = tokens[:max_len]
        sequence = sequence[:max_len]
        
    return torch.tensor([sequence], dtype=torch.long), torch.tensor([actual_length], dtype=torch.long), tokens_padded, actual_length


# Hàm tải cấu hình mô hình và tài nguyên lịch sử huấn luyện
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

    with open(word_map_file, 'r', encoding='utf-8') as f:
        word_map = json.load(f)
        
    with open(metrics_file, 'rb') as f:
        data_pkl = pickle.load(f)
        history_dict = data_pkl['history']
        metrics_data = data_pkl['metrics']

    # Khởi tạo kiến trúc mạng AttBiLSTM
    vocab_size = len(word_map)
    embedding_dim = 256  
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
    
    # Đọc và map bộ trọng số (.pth.tar) vào thiết bị CPU
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

# Thực thi nạp tài nguyên hệ thống
try:
    model, word_map, history_dict, metrics_data = load_all_resources()
except Exception as e:
    st.error(f"Lỗi hệ thống khi tải cấu hình hoặc đọc file tài nguyên: {e}")
    st.stop()

# Giao diện web
st.title(":speech_balloon: Hệ Thống Phân Loại Chủ Đề Văn Bản Yahoo Answers (BiLSTM + Attention)")
st.subheader("**Nhóm 6**")
st.markdown("---")

tab1, tab2 = st.tabs([":crystal_ball: Phân Tích Trực Quan", ":bar_chart: Đánh Giá Hiệu Năng Mô Hình"])

# Tab phân tích văn bản và Attention
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
            st.session_state.actual_len = 0

        if st.button(":rocket: Tiến hành phân tích chủ đề", type="primary", use_container_width=True):
            if user_text.strip() == "":
                st.warning("Vui lòng nhập văn bản trước khi bấm nút dự đoán!")
            else:
                # Gọi hàm tiền xử lý bóc tách chuỗi thực tế
                input_tensor, words_per_sentence, tokens_padded, actual_len = preprocess_text(user_text, word_map)
                
                # Truyền dữ liệu vào mạng nơ-ron thực tế của nhóm
                with torch.no_grad():
                    scores, alphas = model(input_tensor, words_per_sentence, return_attention=True)
                    probabilities = torch.softmax(scores, dim=1).numpy()[0]
                
                pred_class_id = int(np.argmax(probabilities))
                
                # Cập nhật kết quả tính toán thực tế của AI lên UI
                st.session_state.pred_topic = DANH_MỤC_YAHOO[pred_class_id]
                st.session_state.conf_yahoo = probabilities[pred_class_id] * 100
                st.session_state.prob_yahoo = probabilities
                st.session_state.attn_weights = alphas.squeeze(0).numpy()
                st.session_state.tokens = tokens_padded
                st.session_state.actual_len = actual_len

        if st.session_state.prob_yahoo is not None:
            st.markdown("### :bar_chart: Phân phối xác suất các chuyên mục:")
            proba_df = pd.DataFrame({
                'Chuyên mục': list(DANH_MỤC_YAHOO.values()),
                'Xác suất (%)': st.session_state.prob_yahoo * 100
            })
            st.bar_chart(data=proba_df, x='Chuyên mục', y='Xác suất (%)', use_container_width=True)

    with col_phai:
        st.markdown("### :desktop_computer: Kết quả nhận diện hệ thống")
        if st.session_state.pred_topic is None:
            st.info("Nhập đoạn văn bản ở cột bên trái và bấm nút 'Phân tích' để kích hoạt mạng Neural nhận diện!")
            st.markdown("""
            **Gợi ý câu mẫu để test thử:**
            1. *Thể thao:* "Who is the greatest basketball player of all time in NBA history?"
            2. *Khoa học / Toán:* "Can someone explain the theory of relativity and quantum mechanics in simple terms?"
            3. *Kinh doanh:* "How do interest rates affect the stock market and inflation?"
            """)
        else:
            st.success(f":tada: Chủ đề dự báo hệ thống: **{st.session_state.pred_topic}**")
            st.metric(label="Độ tin cậy dự đoán chính xác", value=f"{st.session_state.conf_yahoo:.2f}%")
            
            st.markdown("### :mag: Trực quan hóa Trọng số Attention (Word Importance):")
            st.write("Mô hình mạng Neural đang tập trung vào các từ khóa mang tính quyết định để đưa ra chuyên mục.")
            
            if st.session_state.attn_weights is not None and st.session_state.tokens is not None:
                # Xuất đúng số lượng từ thực tế, loại bỏ vùng đệm padding hiển thị biểu đồ
                eff_len = st.session_state.actual_len
                weights_to_show = st.session_state.attn_weights[:eff_len]
                tokens_to_show = st.session_state.tokens[:eff_len]
                
                # Biểu diễn trực quan phân phối trọng số Attention
                fig_attn, ax_attn = plt.subplots(figsize=(7, max(3, eff_len * 0.35)))
                sns.barplot(x=weights_to_show, y=tokens_to_show, palette="viridis", ax=ax_attn)
                ax_attn.set_title("Mức độ tập trung năng lượng cơ chế Attention vào từng từ", fontsize=10, fontweight='bold')
                ax_attn.set_xlabel("Trọng số alpha (α)")
                st.pyplot(fig_attn)

# Tab đánh giá mô hình
with tab2:
    st.markdown("## :chart_with_upwards_trend: Kết Quả Thực Nghiệm Mạng Học Sâu BiLSTM + Attention")
    st.write("Số liệu kiểm thử mô hình thu được trên tập dữ liệu phân loại văn bản Yahoo Answers.")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Độ chính xác tập Kiểm thử (Test Accuracy)", value=f"{metrics_data['test_accuracy']*100:.2f}%")
    with metric_col2:
        st.metric(label="Độ mất mát tập Kiểm thử (Test Loss)", value=f"{metrics_data['test_loss']:.4f}")
    with metric_col3:
        st.metric(label="Kiến trúc mạng", value="BiLSTM + Attention")

    st.markdown("---")
    st.subheader(":chart_with_upwards_trend: Biểu đồ Quá trình Huấn luyện (Training History)")
    
    col_img1, col_img2 = st.columns(2)
    
    # Tải và hiển thị ảnh Accuracy từ Google Drive
    path_acc = "training_accuracy.png"
    if not os.path.exists(path_acc):
        with st.spinner("Đang tải sơ đồ Accuracy từ Drive..."):
            drive_id_acc = "1PFaRJ3ale_Vy78B47hA836UMQUPts0oz"
            url_acc = f"https://drive.google.com/uc?id={drive_id_acc}"
            gdown.download(url_acc, path_acc, quiet=True)
            
    with col_img1:
        if os.path.exists(path_acc):
            st.image(path_acc, caption="Đồ thị Accuracy qua các Epoch", use_container_width=True)
        else:
            st.warning("Không thể tải hình ảnh Accuracy từ Google Drive.")

    # Tải và hiển thị ảnh Loss từ Google Drive
    path_loss = "training_loss.png"
    if not os.path.exists(path_loss):
        with st.spinner("Đang tải sơ đồ Loss từ Drive..."):
            drive_id_loss = "16o7kyn2XyWUVdjjhIcuVasracHOG6G7N"
            url_loss = f"https://drive.google.com/uc?id={drive_id_loss}"
            gdown.download(url_loss, path_loss, quiet=True)
            
    with col_img2:
        if os.path.exists(path_loss):
            st.image(path_loss, caption="Đồ thị Loss qua các Epoch", use_container_width=True)
        else:
            st.warning("Không thể tải hình ảnh Loss từ Google Drive.")

    st.markdown("---")
    st.subheader(":jigsaw: Ma trận nhầm lẫn (Confusion Matrix)")
    
    # Tải và hiển thị ảnh Confusion Matrix từ Google Drive
    path_cm = "confusion_matrix.png"
    if not os.path.exists(path_cm):
        with st.spinner("Đang tải sơ đồ Confusion Matrix từ Drive..."):
            drive_id_cm = "1cXk8ISNmQNC5XJaqJpwrBbivkZl00GaF"
            url_cm = f"https://drive.google.com/uc?id={drive_id_cm}"
            gdown.download(url_cm, path_cm, quiet=True)
            
    if os.path.exists(path_cm):
        st.image(path_cm, caption="Ma trận nhầm lẫn thực nghiệm", use_container_width=True)
    else:
        st.warning("Không thể tải hình ảnh Ma trận nhầm lẫn từ Google Drive.")

    st.markdown("---")
    st.subheader(":clipboard: Báo cáo phân loại chi tiết (Classification Report)")
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
        
    st.info("**Chú thích ý nghĩa các chỉ số:**\n"
            "- **Precision (Độ chính xác dự báo):** Trong số các mẫu được hệ thống xếp vào chủ đề này, có bao nhiêu phần trăm là đúng thực tế.\n"
            "- **Recall (Độ phủ/Tỉ lệ tìm sót):** Trong số tất cả các mẫu của chủ đề này có có trong tập kiểm thử, hệ thống đã nhận diện được bao nhiêu phần trăm.\n"
            "- **F1-score:** Chỉ số đánh giá cân bằng giữa cả hai yếu tố trên nhằm phản ánh hiệu năng tổng quát.")
