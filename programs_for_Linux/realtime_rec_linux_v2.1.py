import time
import cv2
import numpy as np
from rknn.api import RKNN

class DigitOCR_RKNN:
    def __init__(self, det_model_path, cls_model_path, rec_model_path):
        """
        初始化OCR引擎
        参数：
            det_model_path: 检测模型路径(.rknn)
            cls_model_path: 方向分类模型路径
            rec_model_path: 识别模型路径
        """
        # 初始化三个模型
        self.detector = self._init_rknn_model(det_model_path)  # 文字检测模型
        self.classifier = self._init_rknn_model(cls_model_path)  # 方向分类模型
        self.recognizer = self._init_rknn_model(rec_model_path)  # 文字识别模型

        # 模型参数配置
        self.det_input_size = (960, 640)   # 检测模型输入尺寸(W,H)
        self.cls_input_size = (192, 48)    # 方向分类输入尺寸(W,H)
        self.rec_input_size = (100, 32)    # 识别模型输入尺寸(W,H)
        self.cls_labels = ['0', '180']     # 方向分类标签
        
        # 阈值参数
        self.det_threshold = 0.3   # 检测置信度阈值
        self.cls_threshold = 0.8   # 方向分类阈值
        self.rec_threshold = 0.5   # 识别置信度阈值

    def _init_rknn_model(self, model_path):
        """加载并初始化 RKNN 模型（适用于 RKNN Toolkit 2.3）"""
        rknn = RKNN(verbose=True)
        
        # 加载预转换的 RKNN 模型
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise ValueError(f"模型加载失败: {model_path}, 错误码: {ret}")
        
        # 初始化运行时环境
        # 请根据您的硬件特性确认 core_mask 的设置，如板卡支持双核 NPU 则使用 RKNN.NPU_CORE_0_1，
        # 如果只有单核 NPU 则可设置为 RKNN.NPU_CORE_0
        ret = rknn.init_runtime(
            target='rk3576',
            core_mask=RKNN.NPU_CORE_0_1
        )
        if ret != 0:
            raise ValueError("运行时环境初始化失败，错误码: {}".format(ret))
        
        return rknn

    def _preprocess_det(self, img):
        """
        检测模型预处理
        将原始图像转换为检测模型输入格式
        参数：
            img: 原始BGR图像(numpy数组)
        返回：
            预处理后的张量(CHW格式)
        """
        # 保持长宽比的缩放
        h, w = img.shape[:2]
        target_size = 480  # 与模型输入尺寸一致
        
        # 计算缩放比例
        scale = min(target_size/w, target_size/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # 创建填充后的画布
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # 转换维度顺序并添加批处理维度
        processed = padded.transpose(2, 0, 1)  # HWC -> CHW
        processed = np.expand_dims(processed, axis=0)  # 添加batch维度
        
        # 归一化处理（根据模型训练时的配置）
        normalized = (processed.astype(np.float32) / 255.0 - 0.5) / 0.5
        return normalized

    def _postprocess_det(self, outputs, scale):
        """
        检测模型后处理
        参数：
            outputs: 模型输出结果
            scale: 原始图像到模型输入的缩放比例
        返回：
            检测框列表([x1,y1,x2,y2])
        """
        boxes = []
        scores = []
        
        # 解析模型输出(假设输出格式为[N,6])
        pred = outputs[0].reshape(-1, 6)  # 转换为二维数组

        for det in pred:
            score = det[4]
            if score < self.det_threshold:
                continue  # 过滤低置信度检测
            
            # 将坐标还原到原始图像尺寸
            box = (det[:4] / scale).astype(int)
            boxes.append(box)
            scores.append(score)

        # 执行非极大值抑制
        keep = self.non_max_suppression(np.array(boxes), np.array(scores))
        return [boxes[i] for i in keep]

    def _preprocess_cls(self, roi):
        """
        方向分类预处理
        参数：
            roi: 检测到的候选区域(BGR图像)
        返回：
            预处理后的张量(CHW格式)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 保持比例调整尺寸
        h, w = gray.shape
        ratio = w / float(h)
        resized_w = int(self.cls_input_size[1] * ratio)  # 固定高度
        resized = cv2.resize(gray, (resized_w, self.cls_input_size[1]))
        
        # 填充到固定宽度
        padded = np.zeros((self.cls_input_size[1], self.cls_input_size[0]), dtype=np.uint8)
        padded[:, :resized_w] = resized
        
        # 归一化处理
        normalized = (padded / 255.0).astype(np.float32)
        return np.transpose(np.expand_dims(normalized, axis=2), [2, 0, 1])  # HWC -> CHW

    def _postprocess_cls(self, output):
        """
        方向分类后处理
        参数：
            output: 模型输出结果
        返回：
            方向标签('0'或'180')
        """
        probs = np.squeeze(output)  # 去除冗余维度
        pred_idx = np.argmax(probs)
        return self.cls_labels[pred_idx] if probs[pred_idx] > self.cls_threshold else '0'

    def _preprocess_rec(self, roi):
        """
        识别模型预处理
        参数：
            roi: 方向校正后的区域图像
        返回：
            预处理后的张量(NHWC格式)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值二值化
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 调整尺寸并添加通道维度
        resized = cv2.resize(thresh, self.rec_input_size)
        normalized = (resized / 255.0).astype(np.float32)
        return np.expand_dims(normalized, axis=(0, 3))  # 添加batch和channel维度

    def _postprocess_rec(self, output):
        """
        识别结果后处理
        参数：
            output: 识别模型输出
        返回：
            识别出的数字字符串
        """
        # 假设输出形状为[1, 24, 11] (batch, seq_len, num_classes)
        pred = np.argmax(output[0], axis=1)
        
        # 简单CTC解码
        digits = []
        prev = -1
        for p in pred:
            if p != prev and p != 10:  # 假设第10类为空白符
                digits.append(str(p))
            prev = p
        return ''.join(digits) if digits else ''

    def non_max_suppression(self, boxes, scores, iou_threshold=0.5):
        """
        非极大值抑制(NMS)实现
        参数：
            boxes: 检测框数组(N,4)
            scores: 置信度数组(N,)
            iou_threshold: 重叠阈值
        返回：
            保留的检测框索引列表
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算当前框与其他框的交集
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算交集面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h

            # 计算IoU
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # 保留IoU低于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def rotate_roi(self, roi, angle):
        """
        根据方向标签旋转ROI区域
        参数：
            roi: 原始区域图像
            angle: 方向标签('0'或'180')
        返回：
            校正后的图像
        """
        if angle == '180':
            return cv2.rotate(roi, cv2.ROTATE_180)
        return roi

    def process_frame(self, frame):
        """
        完整处理流程：
        1. 检测数字区域
        2. 方向分类
        3. 数字识别
        参数：
            frame: 输入BGR图像
        返回：
            digits: 识别结果列表[{'box': [x1,y1,x2,y2], 'text': str, 'orientation': str}]
        """
        # 步骤1：数字检测
        det_input = self._preprocess_det(frame)
        det_outputs = self.detector.inference(inputs=[det_input])
        scale = min(self.det_input_size[0]/frame.shape[1], self.det_input_size[1]/frame.shape[0])
        boxes = self._postprocess_det(det_outputs, 1/scale)

        digits = []
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue  # 跳过空区域

            # 步骤2：方向分类
            cls_input = self._preprocess_cls(roi)
            cls_output = self.classifier.inference(inputs=[cls_input])
            angle = self._postprocess_cls(cls_output)
            
            # 方向校正
            corrected_roi = self.rotate_roi(roi, angle)

            # 步骤3：数字识别
            rec_input = self._preprocess_rec(corrected_roi)
            rec_output = self.recognizer.inference(inputs=[rec_input])
            text = self._postprocess_rec(rec_output)
            
            if text.isdigit():
                digits.append({
                    'box': (x1, y1, x2, y2),
                    'text': text,
                    'orientation': angle
                })

        return digits

if __name__ == "__main__":
    # 初始化OCR引擎
    ocr_engine = DigitOCR_RKNN(
        det_model_path="./ppocrv4_det_rk3576.rknn",
        cls_model_path="./ch_ppocr4_cls_rk3576.rknn",
        rec_model_path="./ppocrv4_rec_rk3576.rknn"
    )

    # 初始化摄像头（使用V4L2驱动）
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                print("视频流中断")
                break

            # 执行OCR处理
            start_time = time.time()
            results = ocr_engine.process_frame(frame)
            proc_time = time.time() - start_time

            # 可视化结果
            display_frame = frame.copy()
            for res in results:
                x1, y1, x2, y2 = res['box']
                
                # 根据方向使用不同颜色绘制
                color = (0, 255, 0) if res['orientation'] == '0' else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # 显示识别结果和方向
                text = f"{res['text']}({res['orientation']}deg)"
                cv2.putText(display_frame, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 显示性能信息
            fps = 1 / proc_time if proc_time > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('RK3576 Digit OCR', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()