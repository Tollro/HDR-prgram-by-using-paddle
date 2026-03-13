import cv2
import numpy as np
from rknn.api import RKNN

class PPOCRv4_Recognizer:
    def __init__(self, model_paths):
        # 模型配置参数
        self.model_config = {
            'det': {
                'input_size': (480, 480),
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375]
            },
            'cls': {
                'input_size': (192, 48),  # WxH
                'mean': [123.675, 116.28, 103.53],  # 假设使用相同参数
                'std': [58.395, 57.12, 57.375]
            },
            'rec': {
                'input_size': (320, 48),  # WxH
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375]
            }
        }
        
        # 初始化NPU模型
        self.det_rknn = self._init_model(model_paths['det'], core=RKNN.NPU_CORE_0)
        self.cls_rknn = self._init_model(model_paths['cls'], core=RKNN.NPU_CORE_1)
        self.rec_rknn = self._init_model(model_paths['rec'], core=RKNN.NPU_CORE_0)
        
        # 字符字典（根据实际训练字典调整）
        self.char_dict = ['0','1','2','3','4','5','6','7','8','9']

    def _init_model(self, model_path, core):
        """统一模型初始化方法"""
        rknn = RKNN()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载模型失败: {model_path}")
        
        ret = rknn.init_runtime(
            target='rk3576',
            core_mask=core,
            device_id='110b'
        )
        if ret != 0:
            raise RuntimeError(f"初始化模型失败: {model_path}")
        return rknn

    def _preprocess(self, img, model_type):
        """标准化预处理"""
        cfg = self.model_config[model_type]
        h, w = cfg['input_size']
        
        # 调整尺寸
        img = cv2.resize(img, (w, h))  # OpenCV使用(width, height)
        
        # 颜色通道转换
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化处理
        img = img.astype(np.float32)
        img -= np.array(cfg['mean'], dtype=np.float32)
        img /= np.array(cfg['std'], dtype=np.float32)
        
        # 调整维度顺序
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, 0)  # 添加batch维度

    def detect_text(self, frame):
        """文本检测"""
        # 预处理
        img_data = self._preprocess(frame, 'det')
        
        # 推理
        outputs = self.det_rknn.inference(inputs=[img_data])
        
        # 后处理（示例，需根据实际模型输出调整）
        boxes = outputs[0][0]  # 假设输出形状为[1, N, 4]
        scores = outputs[1][0] # 置信度输出[1, N]
        
        # 过滤低置信度检测
        valid_idx = scores > 0.5
        return boxes[valid_idx]

    def cls_inference(self, crop_img):
        """方向分类"""
        img_data = self._preprocess(crop_img, 'cls')
        output = self.cls_rknn.inference([img_data])
        return output[0][0] > 0.5  # 假设二分类输出

    def rec_inference(self, crop_img):
        """文字识别"""
        img_data = self._preprocess(crop_img, 'rec')
        output = self.rec_rknn.inference([img_data])
        return self._decode(output[0])

    def _decode(self, preds):
        """解码识别结果"""
        text = []
        for c in np.argmax(preds, axis=1):
            if c != 0 and (not (len(text) > 0 and c == text[-1])):
                text.append(c)
        return ''.join([self.char_dict[i] for i in text])

    def process_frame(self, frame):
        """处理单帧"""
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / 640
        scale_y = orig_h / 480
        
        # 缩放以提升处理速度
        frame = cv2.resize(frame, (640, 480))
        
        # 文本检测
        boxes = self.detect_text(frame)
        
        results = []
        for box in boxes:
            # 转换坐标到原始尺寸
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)
            
            # 裁剪区域有效性检查
            if (x2 - x1 < 5) or (y2 - y1 < 5):
                continue
                
            try:
                crop = frame[y1:y2, x1:x2]
                
                # 方向分类
                if self.cls_inference(crop):
                    crop = cv2.rotate(crop, cv2.ROTATE_180)
                
                # 文字识别
                text = self.rec_inference(crop)
                results.append({'box': (x1,y1,x2,y2), 'text': text})
                
            except Exception as e:
                print(f"处理失败: {str(e)}")
                
        return results

    def realtime_recognize(self):
        """实时识别主循环"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 处理帧并计时
            start = cv2.getTickCount()
            results = self.process_frame(frame)
            infer_time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            
            # 绘制结果
            for res in results:
                x1, y1, x2, y2 = res['box']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, res['text'], (x1,y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            # 显示帧率
            fps = 1.0 / infer_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            cv2.imshow('PPOCRv4 Demo', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_paths = {
        'det': '/home/cat/python_docs/dig_rec/ppocrv4_det_rk3576.rknn',
        'cls': '/home/cat/python_docs/dig_rec/ppocr4_cls_rk3576.rknn',
        'rec': '/home/cat/python_docs/dig_rec/ppocrv4_rec_rk3576.rknn'
    }
    
    recognizer = PPOCRv4_Recognizer(model_paths)
    print("启动实时识别...")
    recognizer.realtime_recognize()