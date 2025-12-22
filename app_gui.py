import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Bổ sung thư viện AI ---
try:
    import torch
    from ultralytics import YOLO, SAM
    import shutil # Thêm thư viện để di chuyển và xóa thư mục
except ImportError:
    messagebox.showerror("Dependency Error", "PyTorch and Ultralytics must be installed. Please run: pip install torch torchvision torchaudio ultralytics opencv-python numpy")
    sys.exit(1)


# --- Configuration ---
YOLO_MODEL_PATH = "best.pt" 
SAM_MODEL_PATH = 'sam2_b.pt'
OUTPUT_IMAGE_NAME = 'output_tumor_segmentation.jpg'

# FIXED OUTPUT DIRECTORY
# Sử dụng 'r' để đảm bảo các dấu gạch chéo ngược được xử lý chính xác trong đường dẫn Windows.
FIXED_OUTPUT_DIR = r'C:\Users\admin\Downloads\Knowledge\YOLOV11\tumor_detect'

# --- Device Detection ---
# Tự động chọn thiết bị: CUDA (nếu có) hoặc CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# --- Core Model Logic: Gói gọn logic của bạn vào một hàm ---

def load_models():
    """Tải mô hình YOLO và SAM."""
    # Kiểm tra đường dẫn mô hình
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}. Please ensure 'best.pt' is in the current working directory.")
    if not os.path.exists(SAM_MODEL_PATH):
        raise FileNotFoundError(f"SAM model not found at: {SAM_MODEL_PATH}. Please ensure 'sam2_b.pt' is in the current working directory.")
        
    print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print(f"Loading SAM model from: {SAM_MODEL_PATH}")
    sam_model = SAM(SAM_MODEL_PATH)
    
    return yolo_model, sam_model 

def run_full_pipeline(yolo_model, sam_model, image_path, output_dir):
    """
    Chạy toàn bộ quy trình YOLO và SAM.
    """
    
    # 1. Chạy YOLO detection
    print("Executing YOLO detection...")
    yolo_results = yolo_model(image_path, verbose=False, device=DEVICE)
    
    # Luôn luôn lấy kết quả đầu tiên vì ta chỉ chạy 1 ảnh
    result = yolo_results[0]
    
    # Kiểm tra xem YOLO có phát hiện được box nào không
    if result.boxes and result.boxes.xyxy.numel() > 0:
        boxes_count = result.boxes.xyxy.numel() // 4
        print(f"YOLO detected {boxes_count} bounding boxes.")
        
        # Lấy TẤT CẢ các boxes được YOLO phát hiện (không lọc class)
        boxes_tensor = result.boxes.xyxy.to(DEVICE)
        
        # 2. Chạy SAM segmentation
        print(f"Executing SAM segmentation based on {boxes_count} box prompts and saving...")
        
        # Thiết lập Ultralytics để lưu kết quả vào thư mục tạm thời
        temp_project_name = 'temp_sam_output'
        sam_output_root = os.path.join(output_dir, temp_project_name)
        
        # Chạy SAM tự động vẽ và lưu ảnh
        sam_model(result.orig_img, 
                  bboxes=boxes_tensor, 
                  verbose=False, 
                  save=True, 
                  project=output_dir, 
                  name=temp_project_name, 
                  exist_ok=True, 
                  device=DEVICE)
        
        # 3. Xử lý file đầu ra
        
        sam_saved_path_1 = os.path.join(sam_output_root, 'predict', 'image0.jpg')
        sam_saved_path_2 = os.path.join(sam_output_root, 'image0.jpg')
        
        final_file_path_to_use = None

        if os.path.exists(sam_saved_path_1):
            final_file_path_to_use = sam_saved_path_1
        elif os.path.exists(sam_saved_path_2):
            final_file_path_to_use = sam_saved_path_2

        if final_file_path_to_use:
            final_output_path = os.path.join(output_dir, OUTPUT_IMAGE_NAME)
            
            # Di chuyển và đổi tên file
            shutil.move(final_file_path_to_use, final_output_path)
            
            # Dọn dẹp thư mục tạm thời
            try:
                # Xóa thư mục 'temp_sam_output'
                shutil.rmtree(sam_output_root, ignore_errors=True)
            except Exception as cleanup_err:
                print(f"Warning: Could not clean up temporary directories: {cleanup_err}")

            return final_output_path
        else:
            # Nếu có boxes nhưng SAM không lưu file ở cả 2 vị trí
            print(f"ERROR: SAM ran but output file not found in expected locations:\n1. {sam_saved_path_1}\n2. {sam_saved_path_2}")
            return None
            
    else:
        # Trường hợp YOLO không tìm thấy bất kỳ box nào
        print("YOLO found 0 bounding boxes for the input image.")
        return None # Không tìm thấy khối u hoặc segmentation thất bại

# --- GUI Application Class ---

class TumorDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Tumor Detection and Segmentation (Simplified)")

        self.input_path = tk.StringVar()
        self.status_var = tk.StringVar(value=f"Ready. Using device: {DEVICE}. Select an image file.")
        
        self.yolo_model = None
        self.sam_model = None

        # Load models once when the application starts
        self.status_var.set("Initializing... Loading models.")
        self.master.update()
        try:
            self.yolo_model, self.sam_model = load_models()
            self.status_var.set(f"Ready. Using device: {DEVICE}. Select an image file.")
        except Exception as e:
            error_msg = f"Error loading models: {e}. Check console for details."
            self.status_var.set(error_msg)
            messagebox.showerror("Model Load Error", error_msg)


        # Widgets
        
        # Input Frame
        self.input_frame = tk.Frame(master, padx=10, pady=10)
        self.input_frame.pack(fill='x')

        self.label = tk.Label(self.input_frame, text="Input Image:")
        self.label.pack(side='left')

        self.path_entry = tk.Entry(self.input_frame, textvariable=self.input_path, width=50)
        self.path_entry.pack(side='left', padx=5, expand=True, fill='x')

        self.browse_button = tk.Button(self.input_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side='left')

        # Control Frame
        self.control_frame = tk.Frame(master, padx=10, pady=10)
        self.control_frame.pack(fill='x')
        
        self.run_button = tk.Button(self.control_frame, text="Run Detection", command=self.run_detection)
        self.run_button.pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(side='bottom', fill='x')

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        if filename:
            self.input_path.set(filename)
            self.status_var.set(f"File selected: {os.path.basename(filename)}. Ready to run on {DEVICE}.")

    def run_detection(self):
        input_image_path = self.input_path.get()
        
        if not input_image_path or not os.path.exists(input_image_path):
            messagebox.showerror("Error", "Please select a valid input image file.")
            return

        if not self.yolo_model or not self.sam_model:
            messagebox.showerror("Error", "Models failed to load. Please check if model files exist and restart the application.")
            return
            
        try:
            self.status_var.set("Processing: Running full pipeline...")
            self.master.update()
            
            # Đảm bảo thư mục đầu ra tồn tại
            os.makedirs(FIXED_OUTPUT_DIR, exist_ok=True)
            
            output_image_path = run_full_pipeline(
                self.yolo_model, 
                self.sam_model, 
                input_image_path, 
                FIXED_OUTPUT_DIR
            )
            
            if output_image_path:
                
                # Tự động mở ảnh (sử dụng os.startfile cho Windows)
                try:
                    os.startfile(output_image_path)
                except Exception as open_err:
                    print(f"Warning: Could not auto-open image: {open_err}")
                
                self.status_var.set(f"SUCCESS: Result saved to {output_image_path} (Device: {DEVICE})")
                messagebox.showinfo("Success", f"Segmentation result saved to:\n{output_image_path}\nRunning on: {DEVICE}")
            else:
                self.status_var.set("Completed: No tumor found.")
                messagebox.showinfo("Result", "No tumor found.")

        except Exception as e:
            self.status_var.set(f"Runtime Error: {e}")
            messagebox.showerror("Error", f"An error occurred during processing: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectorApp(root)
    root.mainloop()