import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
import shutil

class FaceSwapApp:
    def __init__(self):
        self.model = None
        self.target_faces = []  # 存储目标人脸图片
        self.target_face_paths = ['1v.png', '2v.png', '3v.png']  # 目标人脸路径列表
        # 确保使用当前工作目录的绝对路径
        self.target_face_paths = [os.path.join(os.getcwd(), path) for path in self.target_face_paths]
        self.yolo_threshold = 0.5  # YOLOv8的置信度阈值
        self.use_yolo = True  # 强制使用YOLOv8
        self.annotations = []  # 存储手动标注的数据
        self.current_canvas = None  # 存储当前canvas引用
        
    def show_menu(self):
        while True:
            print("\n" + "="*50)
            print(" 面部替换工具 - 主菜单 (仅YOLOv8)")
            print("="*50)
            print("1. 下载并初始化YOLOv8模型")
            print("2. 创建手动标注工具（B站分辨率支持）")
            print("3. 训练YOLOv8模型（使用txt文件夹数据集）")
            print("4. 测试YOLOv8阈值")
            print("5. 开始面部替换工作（使用目标截图）")
            print("6. 设置YOLOv8阈值")
            print("7. 退出")
            print("="*50)
            
            choice = input("请输入选项 (1-7): ")
            
            if choice == "1":
                self.download_and_initialize_yolo()
            elif choice == "2":
                self.create_manual_annotation_gui()
            elif choice == "3":
                self.train_yolo_with_txt_folder()
            elif choice == "4":
                self.test_yolo_threshold()
            elif choice == "5":
                self.start_face_swap()
            elif choice == "6":
                self.set_yolo_threshold()
            elif choice == "7":
                print("退出程序...")
                break
            else:
                print("无效选项，请重新输入")

    def download_and_initialize_yolo(self):
        print("\n正在下载和初始化YOLOv8模型...")
        
        # 下载YOLOv8s模型
        print("1. 下载YOLOv8s模型...")
        try:
            self.model = YOLO('yolov8s.pt')
            print("✅ YOLOv8s模型加载成功")
        except Exception as e:
            print(f"❌ YOLOv8s模型加载失败: {e}")
            print("尝试从Hugging Face下载...")
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id="ultralytics/yolov8s", filename="yolov8s.pt")
                self.model = YOLO(model_path)
                print("✅ 从Hugging Face成功下载YOLOv8s")
            except Exception as e:
                print(f"❌ 从Hugging Face下载也失败: {e}")
                print("请手动下载YOLOv8s模型: https://github.com/ultralytics/assets/releases")
                return
        
        # 加载目标人脸图片
        print("\n2. 加载目标人脸图片...")
        self.target_faces = []
        for path in self.target_face_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self.target_faces.append(img)
                    print(f"  - 成功加载目标图片: {path}")
                else:
                    print(f"  - 警告: 无法读取目标图片 {path}")
            else:
                print(f"  - 警告: 目标图片 {path} 不存在")
        
        if len(self.target_faces) == 0:
            print("❌ 没有找到有效的目标人脸图片")
            return
        
        print("\n✅ YOLOv8模型和目标图片初始化完成！")

    def create_manual_annotation_gui(self):
        print("\n创建手动标注工具（B站分辨率支持）...")
        
        root = tk.Tk()
        root.title("手动面部标注工具")
        root.geometry("1200x800")
        
        canvas = tk.Canvas(root, width=800, height=600, scrollregion=(0, 0, 1920, 1080))
        canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.current_canvas = canvas
        
        scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def select_file():
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            if file_path:
                img = cv2.imread(file_path)
                if img is not None:
                    self.display_image_for_annotation(canvas, img, root, file_path)
        
        btn_select = tk.Button(root, text="选择图片进行手动标注", command=select_file)
        btn_select.pack(pady=5)
        
        def save_annotations():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w') as f:
                    for annotation in self.annotations:
                        f.write(f"{annotation['file']},{annotation['x1']},{annotation['y1']},{annotation['x2']},{annotation['y2']},{annotation['label']}\n")
                messagebox.showinfo("保存成功", f"标注数据已保存到: {file_path}")
        
        btn_save = tk.Button(root, text="保存标注数据", command=save_annotations)
        btn_save.pack(pady=5)
        
        def clear_annotations():
            self.annotations = []
            messagebox.showinfo("清除成功", "所有标注数据已清除")
        
        btn_clear = tk.Button(root, text="清除标注", command=clear_annotations)
        btn_clear.pack(pady=5)
        
        self.annotation_mode = tk.BooleanVar(value=False)
        chk_annotation = tk.Checkbutton(root, text="启用手动标注模式", variable=self.annotation_mode)
        chk_annotation.pack(pady=5)
        
        tk.Label(root, text="标注标签 (如: OTTO):").pack(pady=5)
        self.annotation_label = tk.Entry(root)
        self.annotation_label.pack(pady=5)
        self.annotation_label.insert(0, "OTTO")
        
        root.mainloop()

    def display_image_for_annotation(self, canvas, img, root, file_path):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        canvas_width = 800
        canvas_height = 600
        
        img_width, img_height = pil_img.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        tk_img = ImageTk.PhotoImage(pil_img)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img
        
        self.current_image = {
            'path': file_path,
            'pil_img': pil_img,
            'tk_img': tk_img,
            'original_size': (img_width, img_height)
        }
        
        canvas.bind("<Button-1>", lambda e: self.start_annotation(e))
        canvas.bind("<B1-Motion>", lambda e: self.update_annotation(e))
        canvas.bind("<ButtonRelease-1>", lambda e: self.end_annotation(e))
        
        root.update()

    def start_annotation(self, event):
        if not self.annotation_mode.get():
            return
        self.annotation_start = (event.x, event.y)
        self.annotation_rect = self.current_canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2
        )

    def update_annotation(self, event):
        if not self.annotation_mode.get() or not hasattr(self, 'annotation_rect'):
            return
        self.current_canvas.coords(self.annotation_rect, 
                    self.annotation_start[0], self.annotation_start[1],
                    event.x, event.y)

    def end_annotation(self, event):
        if not self.annotation_mode.get() or not hasattr(self, 'annotation_rect'):
            return
        
        coords = self.current_canvas.coords(self.annotation_rect)
        x1, y1, x2, y2 = coords
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        label = self.annotation_label.get() or "unknown"
        
        annotation = {
            'file': self.current_image['path'],
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'label': label
        }
        self.annotations.append(annotation)
        
        print(f"添加标注: {annotation}")
        messagebox.showinfo("标注成功", f"已添加标注: ({x1}, {y1}) - ({x2}, {y2}) 标签: {label}")

    def train_yolo_with_txt_folder(self):
        print("\nYOLOv8训练功能（使用txt文件夹数据集）...")
        
        if self.model is None:
            print("模型未加载，正在自动加载YOLOv8s模型...")
            try:
                self.model = YOLO('yolov8s.pt')
                print("✅ YOLOv8s模型加载成功")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print("请先运行选项1初始化模型")
                return
        
        txt_folder = os.path.join(os.getcwd(), "txt")  # 使用当前工作目录的绝对路径
        print(f"检查txt文件夹: {os.path.abspath(txt_folder)}")
        
        if not os.path.exists(txt_folder):
            print(f"❌ txt文件夹不存在: {os.path.abspath(txt_folder)}")
            print("请确保在根目录下有txt文件夹")
            return
        
        try:
            all_files = os.listdir(txt_folder)
            print(f"txt文件夹中的所有文件: {all_files}")
            
            png_files = [f for f in all_files if f.endswith('.png')]
            txt_files = [f for f in all_files if f.endswith('.txt')]
            
            print(f"找到 {len(png_files)} 个png文件: {png_files}")
            print(f"找到 {len(txt_files)} 个txt文件: {txt_files}")
            
            if len(png_files) == 0 or len(txt_files) == 0:
                print("❌ txt文件夹中没有找到png或txt文件")
                return
            
        except Exception as e:
            print(f"❌ 读取txt文件夹时出错: {e}")
            return
        
        print(f"✅ 找到 {len(png_files)} 个png文件和 {len(txt_files)} 个txt文件")
        
        temp_dataset_path = os.path.join(os.getcwd(), "temp_yolo_dataset")  # 使用当前工作目录的绝对路径
        if os.path.exists(temp_dataset_path):
            shutil.rmtree(temp_dataset_path)
        
        os.makedirs(temp_dataset_path, exist_ok=True)
        os.makedirs(os.path.join(temp_dataset_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(temp_dataset_path, "labels"), exist_ok=True)
        
        for png_file in png_files:
            src_path = os.path.join(txt_folder, png_file)
            dst_path = os.path.join(temp_dataset_path, "images", png_file)
            
            try:
                shutil.copy2(src_path, dst_path)
                print(f"复制图片: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"❌ 复制图片失败: {e}")
                continue
            
            img = cv2.imread(src_path)
            if img is None:
                print(f"❌ 无法读取图片: {src_path}")
                continue
            img_h, img_w = img.shape[:2]
            
            txt_file = png_file.replace('.png', '.txt')
            if txt_file in txt_files:
                src_txt_path = os.path.join(txt_folder, txt_file)
                dst_txt_path = os.path.join(temp_dataset_path, "labels", txt_file)
                
                try:
                    with open(src_txt_path, 'r') as f:
                        lines = f.readlines()
                    
                    print(f"处理标注文件: {src_txt_path}")
                    
                    yolo_lines = []
                    for i, line in enumerate(lines):
                        parts = line.strip().split(',')
                        
                        coords = []
                        for p in parts:
                            try:
                                coords.append(float(p))
                            except ValueError:
                                continue
                        
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = coords[:4]
                            
                            x_center = (x1 + x2) / 2 / img_w
                            y_center = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    with open(dst_txt_path, 'w') as f:
                        f.writelines(yolo_lines)
                    print(f"✅ 转换标注文件: {dst_txt_path} (转换了 {len(yolo_lines)} 个标注)")
                    
                except Exception as e:
                    print(f"❌ 处理标注文件失败: {e}")
        
        print(f"✅ 已创建临时YOLO数据集: {temp_dataset_path}")
        
        # ========== 关键修复：创建data.yaml文件 ==========
        data_yaml = os.path.join(temp_dataset_path, "data.yaml")
        
        # 获取绝对路径，并将反斜杠转换为正斜杠（YOLOv8推荐格式）
        abs_dataset_path = os.path.abspath(temp_dataset_path).replace('\\', '/')
        
        with open(data_yaml, 'w', encoding='utf-8') as f:
            f.write(f"""path: {abs_dataset_path}
train: images
val: images
test: images

names:
  0: OTTO
""")
        
        print(f"✅ 创建data.yaml文件: {data_yaml}")
        print(f"   - 数据集绝对路径: {abs_dataset_path}")
        
        print("\n正在启动YOLOv8训练...")
        print("训练可能需要较长时间，请耐心等待...")
        
        try:
            results = self.model.train(
                data=data_yaml,
                epochs=50,
                imgsz=640,
                batch=16,
                name="custom_face_detection"
            )
            
            print("\n✅ YOLOv8训练完成！")
            print(f"训练结果保存在: runs/detect/custom_face_detection/")
            
            trained_model_path = os.path.join(os.getcwd(), "trained_yolov8s_custom.pt")  # 使用当前工作目录的绝对路径
            self.model.save(trained_model_path)
            print(f"✅ 训练好的模型已保存到: {trained_model_path}")
            
        except Exception as e:
            print(f"\n❌ 训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            print("请检查:")
            print("1. 是否有GPU支持")
            print("2. 数据集格式是否正确")
            print("3. 路径权限是否正确")

    def test_yolo_threshold(self):
        print("\n测试YOLOv8阈值...")
        print("请选择要测试的图片:")
        
        test_img_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not test_img_path:
            print("❌ 未选择测试图片")
            return
        
        img = cv2.imread(test_img_path)
        if img is None:
            print("❌ 无法读取图片")
            return
        
        print(f"\n当前YOLOv8置信度阈值: {self.yolo_threshold}")
        print(f"测试图片: {test_img_path}")
        print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
        
        if self.model:
            # 不使用阈值过滤，获取所有检测结果
            results = self.model(img, classes=[0], conf=0.0)  # 设置conf=0.0获取所有检测结果
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                print(f"\n总共检测到 {len(results[0].boxes)} 个可能的人脸区域:")
                print("="*60)
                
                # 按置信度从高到低排序
                boxes = sorted(results[0].boxes, key=lambda x: x.conf[0], reverse=True)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    
                    # 判断是否超过当前阈值
                    above_threshold = confidence > self.yolo_threshold
                    status = "✅ 通过" if above_threshold else "❌ 低于阈值"
                    
                    print(f"\n人脸区域 {i+1} {status}:")
                    print(f"  - 置信度(相似度): {confidence:.4f} ({confidence*100:.2f}%)")
                    print(f"  - 边界框: ({x1}, {y1}) 到 ({x2}, {y2})")
                    print(f"  - 宽度: {x2-x1}px, 高度: {y2-y1}px")
                    print(f"  - 当前阈值: {self.yolo_threshold}")
                
                # 统计超过阈值的数量
                above_threshold_count = sum(1 for box in boxes if box.conf[0] > self.yolo_threshold)
                print(f"\n" + "="*60)
                print(f"总结: {above_threshold_count}/{len(boxes)} 个人脸区域超过当前阈值 {self.yolo_threshold}")
                
                # 如果有超过阈值的，显示最高置信度
                if above_threshold_count > 0:
                    max_confidence = max(box.conf[0] for box in boxes if box.conf[0] > self.yolo_threshold)
                    print(f"最高置信度: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
            else:
                print("\n❌ 未检测到任何可能的人脸区域")
                print("可能的原因:")
                print("1. 图片中确实没有人脸")
                print("2. 人脸太小或太模糊")
                print("3. 光线条件不佳")
                print("4. 人脸角度不常见")
                
                # 提供调整建议
                if self.yolo_threshold > 0.3:
                    print(f"\n💡 建议: 当前阈值较高({self.yolo_threshold})，尝试降低阈值可能会检测到更多人脸")
                    print("   可以使用选项6调整YOLOv8阈值")
        else:
            print("❌ 模型未加载，请先运行选项1初始化模型")

    def start_face_swap(self):
        print("\n开始面部替换工作（使用目标截图）...")
        
        # 检查是否有有效的目标人脸图片
        if len(self.target_faces) == 0:
            print("⚠️ 未找到有效的目标人脸图片")
            choice = input("请选择操作:\n1. 手动选择目标人脸图片\n2. 返回主菜单\n请输入选项 (1-2): ")
            
            if choice == "1":
                self.load_target_faces()
                if len(self.target_faces) == 0:
                    print("❌ 未能加载任何目标人脸图片，返回主菜单")
                    return
            else:
                print("返回主菜单")
                return
        
        # 显示当前加载的目标人脸图片信息
        print(f"\n当前已加载 {len(self.target_faces)} 张目标人脸图片:")
        for i, face_path in enumerate(self.target_face_paths):
            if i < len(self.target_faces):
                print(f"  {i+1}. {os.path.basename(face_path)}")
        
        # 询问是否需要添加更多目标人脸图片
        add_more = input("\n是否需要添加更多目标人脸图片? (y/n): ").lower()
        if add_more == 'y':
            self.load_target_faces()
        
        # 选择视频文件
        print("\n请选择要处理的视频文件:")
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        
        if not video_path:
            print("❌ 未选择视频文件")
            return
        
        # 选择输出路径
        print("\n请选择输出视频的保存位置:")
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if not output_path:
            print("❌ 未选择输出路径")
            return
        
        # 显示处理信息并开始处理
        print(f"\n准备开始处理:")
        print(f"  - 输入视频: {video_path}")
        print(f"  - 输出视频: {output_path}")
        print(f"  - 当前YOLOv8阈值: {self.yolo_threshold}")
        print(f"  - 目标人脸图片数量: {len(self.target_faces)}")
        
        # 询问是否需要调整阈值
        print(f"\n当前YOLOv8置信度阈值为: {self.yolo_threshold}")
        print("阈值越高，检测越严格但可能漏掉一些人脸")
        print("阈值越低，检测越宽松但可能误检")
        adjust = input("是否需要调整YOLOv8阈值? (y/n): ").lower()
        if adjust == 'y':
            try:
                new_threshold = float(input(f"请输入新的阈值 (0.0-1.0, 当前: {self.yolo_threshold}): "))
                if 0.0 <= new_threshold <= 1.0:
                    self.yolo_threshold = new_threshold
                    print(f"✅ 阈值已调整为: {self.yolo_threshold}")
                else:
                    print(f"❌ 输入无效，保持原阈值: {self.yolo_threshold}")
            except ValueError:
                print(f"❌ 输入无效，保持原阈值: {self.yolo_threshold}")
        
        # 询问是否需要先测试阈值
        test_threshold = input("\n是否需要先测试当前阈值效果? (y/n): ").lower()
        if test_threshold == 'y':
            self.test_yolo_threshold_on_video(video_path)
        
        confirm = input("\n确认开始处理? (y/n): ").lower()
        if confirm != 'y':
            print("已取消处理")
            return
        
        print("\n开始处理视频...")
        self.process_video(video_path, output_path)
    
    def test_yolo_threshold_on_video(self, video_path):
        """在视频上测试当前阈值效果"""
        print(f"\n正在视频 {video_path} 上测试阈值效果...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 无法打开视频文件")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        test_frames = min(10, total_frames)  # 先测试10帧进行详细分析
        frame_step = max(1, total_frames // test_frames)
        
        detection_count = 0
        total_detections = 0
        all_detections = []  # 存储所有检测结果（包括低于阈值的）
        
        print(f"将详细分析 {test_frames} 帧来评估检测效果...")
        
        for i in range(test_frames):
            frame_idx = i * frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            print(f"\n分析帧 {frame_idx}:")
            
            if self.model:
                # 获取所有检测结果，不使用阈值过滤
                results = self.model(frame, classes=[0], conf=0.0)
                
                frame_detections = 0
                frame_all_detections = 0
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        confidence = box.conf[0]
                        frame_all_detections += 1
                        all_detections.append(confidence)
                        
                        print(f"  - 检测到人脸，置信度: {confidence:.4f}")
                        
                        if confidence > self.yolo_threshold:
                            frame_detections += 1
                            total_detections += 1
                
                if frame_all_detections == 0:
                    print("  - 未检测到任何人脸区域")
                    print("  可能的原因:")
                    print("    1. 视频中确实没有人脸")
                    print("    2. 人脸太小或太模糊")
                    print("    3. 光线条件不佳")
                    print("    4. 人脸角度不常见")
                    print("    5. 模型未正确加载")
                else:
                    detection_count += 1
                    print(f"  - 超过当前阈值 {self.yolo_threshold} 的检测数: {frame_detections}")
        
        cap.release()
        
        print(f"\n详细检测结果:")
        print(f"  - 测试帧数: {test_frames}")
        print(f"  - 有人脸的帧数: {detection_count}")
        print(f"  - 总检测次数: {total_detections}")
        print(f"  - 总检测区域数（包括低置信度）: {len(all_detections)}")
        
        if len(all_detections) > 0:
            print(f"  - 最高置信度: {max(all_detections):.4f}")
            print(f"  - 最低置信度: {min(all_detections):.4f}")
            print(f"  - 平均置信度: {sum(all_detections)/len(all_detections):.4f}")
            
            # 提供阈值建议
            print(f"\n💡 阈值调整建议:")
            if max(all_detections) < self.yolo_threshold:
                print(f"  - 当前阈值 {self.yolo_threshold} 高于最高检测置信度 {max(all_detections):.4f}")
                print(f"  - 建议将阈值设置为: {max(all_detections) * 0.8:.4f}")
            else:
                high_conf_count = sum(1 for c in all_detections if c > 0.5)
                if high_conf_count > 0:
                    print(f"  - 有 {high_conf_count} 个高置信度检测 (>0.5)，当前阈值可能合适")
                else:
                    print(f"  - 所有检测置信度都较低，建议检查视频质量或使用更低阈值")
                    print(f"  - 建议将阈值设置为: {max(all_detections) * 0.7:.4f}")
        else:
            print(f"\n⚠️ 未检测到任何可能的人脸区域")
            print("可能的原因:")
            print("1. 视频中确实没有人脸")
            print("2. 模型未正确加载或损坏")
            print("3. 视频格式不支持或损坏")
            print("4. 人脸太小（小于模型最小检测尺寸）")
            
            # 提供进一步诊断选项
            diagnose = input("\n是否需要进行进一步诊断? (y/n): ").lower()
            if diagnose == 'y':
                self.diagnose_model_and_video(video_path)
        
        input("\n按回车键继续...")
    
    def diagnose_model_and_video(self, video_path):
        """进一步诊断模型和视频问题"""
        print("\n=== 详细诊断 ===")
        
        # 检查模型状态
        print(f"1. 检查模型状态:")
        if self.model is None:
            print("   ❌ 模型未加载")
            return
        else:
            print("   ✅ 模型已加载")
        
        # 检查视频基本信息
        print(f"\n2. 检查视频信息:")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("   ❌ 无法打开视频文件")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   - 分辨率: {width}x{height}")
        print(f"   - 帧率: {fps}")
        print(f"   - 总帧数: {total_frames}")
        
        # 尝试读取第一帧
        ret, frame = cap.read()
        if not ret:
            print("   ❌ 无法读取视频帧")
            cap.release()
            return
        
        print("   ✅ 可以正常读取视频帧")
        
        # 保存第一帧作为测试图片
        test_frame_path = os.path.join(os.getcwd(), "test_frame.jpg")
        cv2.imwrite(test_frame_path, frame)
        print(f"   ✅ 已保存测试帧到: {test_frame_path}")
        
        cap.release()
        
        # 使用模型测试第一帧
        print(f"\n3. 使用模型测试第一帧:")
        try:
            results = self.model(frame, conf=0.0)  # 不使用类别过滤，检测所有对象
            print(f"   - 模型推理成功")
            
            total_objects = 0
            for result in results:
                boxes = result.boxes
                total_objects += len(boxes)
                
                for box in boxes:
                    cls = int(box.cls[0])
                    confidence = box.conf[0]
                    class_name = self.model.names[cls] if hasattr(self.model, 'names') else f"Class {cls}"
                    print(f"   - 检测到: {class_name}, 置信度: {confidence:.4f}")
            
            if total_objects == 0:
                print("   ⚠️ 模型未检测到任何对象")
                print("   可能是模型问题或视频中确实没有明显对象")
            else:
                print(f"   ✅ 模型正常工作，共检测到 {total_objects} 个对象")
                
                # 检查是否检测到人脸（类别0）
                person_detections = sum(1 for result in results for box in result.boxes if int(box.cls[0]) == 0)
                if person_detections == 0:
                    print("   ⚠️ 未检测到人脸类别，但检测到其他对象")
                    print("   可能是视频中确实没有人脸，或者人脸太小/不清晰")
        
        except Exception as e:
            print(f"   ❌ 模型推理失败: {e}")
        
        print(f"\n4. 建议的解决方案:")
        print("   1. 检查测试帧图片，确认视频中是否有人脸")
        print("   2. 如果有人脸但未检测到，尝试降低阈值到0.1或更低")
        print("   3. 确保人脸在画面中足够大（建议最小32x32像素）")
        print("   4. 尝试使用其他视频测试")
        print("   5. 如果问题持续，可能需要重新训练或下载模型")
    
    def load_target_faces(self):
        """加载目标人脸图片"""
        print("\n请选择目标人脸图片 (可多选):")
        
        file_paths = filedialog.askopenfilenames(
            title="选择目标人脸图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not file_paths:
            print("未选择任何图片")
            return
        
        loaded_count = 0
        for file_path in file_paths:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    self.target_faces.append(img)
                    self.target_face_paths.append(file_path)
                    loaded_count += 1
                    print(f"✅ 成功加载: {os.path.basename(file_path)}")
                else:
                    print(f"❌ 无法读取: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ 加载失败 {os.path.basename(file_path)}: {e}")
        
        print(f"\n总共加载了 {loaded_count} 张目标人脸图片")
        print(f"当前共有 {len(self.target_faces)} 张目标人脸图片可供替换")

    def process_video(self, video_path, output_path):
        print(f"\n正在打开视频文件: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 无法打开视频文件，请检查文件是否损坏或格式是否支持")
            return
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息:")
        print(f"  - 分辨率: {width}x{height}")
        print(f"  - 帧率: {fps:.2f} FPS")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ 无法创建输出视频文件，请检查输出路径权限")
            cap.release()
            return
        
        frame_idx = 0
        swap_count = 0
        no_detection_count = 0
        max_no_detection = 100  # 连续100帧没有检测到人脸时提示
        
        print("\n开始处理视频帧...")
        print("提示: 按 Ctrl+C 可以中断处理")
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # 每处理一定帧数显示进度
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_idx / total_frames) * 100
                    eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
                    print(f"处理进度: {frame_idx}/{total_frames} 帧 ({progress:.1f}%) | "
                          f"已换脸: {swap_count} 次 | 耗时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s")
                
                if self.model:
                    # 使用YOLOv8检测人脸
                    # 注意：YOLOv8默认不直接支持人脸检测，我们使用person检测然后定位人脸区域
                    results = self.model(frame, classes=[0], conf=0.0)  # 检测所有person类别
                    
                    detected_faces = 0
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = box.conf[0]
                            
                            # 只处理超过阈值的人体检测
                            if confidence > self.yolo_threshold:
                                # 从人体区域估算人脸位置
                                # 人脸通常在人体上部1/4区域，且居中
                                body_height = y2 - y1
                                body_width = x2 - x1
                                
                                # 估算人脸区域（人体上部1/4，宽度居中1/2）
                                face_y1 = y1
                                face_y2 = y1 + int(body_height * 0.35)  # 人脸约占身体高度的35%
                                face_height = face_y2 - face_y1
                                
                                # 人脸宽度约为高度的80%，居中放置
                                face_width = int(face_height * 0.8)
                                face_x1 = x1 + (body_width - face_width) // 2
                                face_x2 = face_x1 + face_width
                                
                                # 确保人脸区域在人体边界内
                                face_x1 = max(x1, face_x1)
                                face_x2 = min(x2, face_x2)
                                face_y2 = min(y1 + int(body_height * 0.5), face_y2)  # 不超过身体上部50%
                                
                                # 跳过太小的人脸区域
                                if (face_x2 - face_x1) < 30 or (face_y2 - face_y1) < 30:
                                    print(f"跳过太小的人脸区域: {(face_x2-face_x1)}x{(face_y2-face_y1)}")
                                    continue
                                
                                detected_faces += 1
                                
                                # 选择目标人脸图片（循环使用）
                                target_idx = swap_count % len(self.target_faces)
                                target_img = self.target_faces[target_idx]
                                
                                # 调整目标图片大小以匹配估算的人脸区域
                                try:
                                    resized_target = cv2.resize(target_img, (face_x2 - face_x1, face_y2 - face_y1))
                                    
                                    # 应用边缘融合使替换更自然
                                    if (face_x2 - face_x1) > 60 and (face_y2 - face_y1) > 60:  # 只对较大的人脸应用融合
                                        # 创建矩形掩码用于边缘融合，保持原始图片形状
                                        mask = np.ones((face_y2 - face_y1, face_x2 - face_x1), dtype=np.float32)
                                        
                                        # 创建边缘渐变效果，使替换更自然
                                        edge_width = min(15, min(face_x2 - face_x1, face_y2 - face_y1) // 4)  # 边缘宽度为较小边长的1/4，最大15像素
                                        
                                        # 上边缘
                                        for i in range(edge_width):
                                            mask[i, :] = i / edge_width
                                        # 下边缘
                                        for i in range(edge_width):
                                            mask[(face_y2 - face_y1) - 1 - i, :] = i / edge_width
                                        # 左边缘
                                        for i in range(edge_width):
                                            mask[:, i] = np.maximum(mask[:, i], i / edge_width)
                                        # 右边缘
                                        for i in range(edge_width):
                                            mask[:, (face_x2 - face_x1) - 1 - i] = np.maximum(mask[:, (face_x2 - face_x1) - 1 - i], i / edge_width)
                                        
                                        # 应用高斯模糊使边缘更平滑
                                        cv2.GaussianBlur(mask, (15, 15), 0)
                                        
                                        # 应用掩码，只替换人脸区域
                                        for c in range(3):
                                            frame[face_y1:face_y2, face_x1:face_x2, c] = (
                                                frame[face_y1:face_y2, face_x1:face_x2, c] * (1 - mask) + 
                                                resized_target[:, :, c] * mask
                                            )
                                    else:
                                        # 小人脸直接替换
                                        frame[face_y1:face_y2, face_x1:face_x2] = resized_target
                                    
                                    swap_count += 1
                                    
                                    # 在帧上绘制替换信息（可选）
                                    cv2.putText(frame, f"Face {swap_count} (Conf: {confidence:.2f})", 
                                               (face_x1, face_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    
                                    # 绘制人脸区域边界框（调试用，可选）
                                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 255), 1)
                                    
                                except Exception as e:
                                    print(f"处理人脸时出错: {e}")
                                    continue
                    
                    # 统计没有检测到人脸的帧数
                    if detected_faces == 0:
                        no_detection_count += 1
                        if no_detection_count >= max_no_detection:
                            print(f"⚠️ 已连续 {no_detection_count} 帧未检测到人脸，可能需要调整阈值")
                            no_detection_count = 0  # 重置计数器，避免重复提示
                    else:
                        no_detection_count = 0
                
                # 写入处理后的帧
                out.write(frame)
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断处理")
            print(f"已处理 {frame_idx}/{total_frames} 帧")
        
        # 释放资源
        cap.release()
        out.release()
        
        # 显示处理结果
        elapsed = time.time() - start_time
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ 视频处理完成！")
        print(f"   - 总帧数: {frame_idx}/{total_frames}")
        print(f"   - 换脸次数: {swap_count}")
        print(f"   - 总耗时: {elapsed:.1f}秒 ({elapsed/60:.2f}分钟)")
        print(f"   - 平均处理速度: {avg_fps:.2f} FPS")
        print(f"   - 输出文件: {output_path}")
        
        # 检查输出文件是否成功创建
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ 输出视频文件已成功创建")
            
            # 询问是否播放视频
            play = input("\n是否播放处理后的视频? (y/n): ").lower()
            if play == 'y':
                self.play_video(output_path)
        else:
            print(f"❌ 输出视频文件创建失败")
    
    def play_video(self, video_path):
        """使用系统默认播放器播放视频"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(video_path)
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{video_path}"' if sys.platform == 'darwin' else f'xdg-open "{video_path}"')
            print(f"正在播放视频: {video_path}")
        except Exception as e:
            print(f"无法播放视频: {e}")

    def set_yolo_threshold(self):
        print("\n设置YOLOv8阈值...")
        print(f"当前YOLOv8置信度阈值: {self.yolo_threshold}")
        
        try:
            new_threshold = float(input("请输入新的YOLOv8置信度阈值 (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                self.yolo_threshold = new_threshold
                print(f"✅ YOLOv8阈值已设置为: {self.yolo_threshold}")
            else:
                print("❌ 阈值必须在0.0到1.0之间")
        except ValueError:
            print("❌ 请输入有效的数字")

if __name__ == "__main__":
    app = FaceSwapApp()
    app.show_menu()