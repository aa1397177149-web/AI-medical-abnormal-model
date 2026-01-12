import os
import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import time
from werkzeug.serving import WSGIRequestHandler
from typing import Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
from functools import wraps
import json
import requests
from datetime import datetime
import threading

# =================配置区域=================
@dataclass
class Config:
    """统一配置管理"""
    # 模型配置
    MODEL_PATH: str = os.environ.get('MODEL_PATH', 'best_medical_model.pth')
    BERT_MODEL_NAME: str = os.environ.get('BERT_MODEL_NAME', 'trueto/medbert-base-wwm-chinese')
    
    # 服务器配置
    PORT: int = int(os.environ.get('PORT', 6006))
    HOST: str = os.environ.get('HOST', '0.0.0.0')
    DEBUG: bool = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # 日志配置
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE: str = 'logs/server.log'
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # 模型配置
    MAX_TEXT_LENGTH: int = 128
    REQUEST_TIMEOUT: int = 30
    BATCH_MAX_SIZE: int = 50
    
    # Ollama配置
    OLLAMA_HOST: str = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_TIMEOUT: int = 180
    OLLAMA_STREAM_CHUNK_SIZE: int = 1024
    
    # CORS配置
    CORS_ORIGINS: str = "*"
    CORS_METHODS: list = None
    CORS_HEADERS: list = None
    
    # 重试配置
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    def __post_init__(self):
        if self.CORS_METHODS is None:
            self.CORS_METHODS = ["GET", "POST", "OPTIONS"]
        if self.CORS_HEADERS is None:
            self.CORS_HEADERS = ["Content-Type", "Authorization", "Accept"]
        
        # 创建日志目录
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)

config = Config()

# =================日志配置=================
class LoggerSetup:
    """增强的日志系统配置"""
    
    @staticmethod
    def setup_logger(name: str, level: str = 'INFO', log_file: str = 'server.log') -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        if logger.handlers:
            return logger
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        # 文件处理器（带日志轮转）
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            logger.info(f"日志文件已创建: {log_file}")
        except Exception as e:
            logger.warning(f"无法创建日志文件: {e}")
        
        return logger

logger = LoggerSetup.setup_logger("MedicalAI", config.LOG_LEVEL, config.LOG_FILE)

# =================数据模型=================
class PredictionLabel(str, Enum):
    """预测标签枚举"""
    ABNORMAL = "abnormal"
    NORMAL = "normal"
    REVIEW = "review"
    
    @property
    def chinese(self) -> str:
        mapping = {
            self.ABNORMAL: "异常",
            self.NORMAL: "正常",
            self.REVIEW: "需复查"
        }
        return mapping[self]

@dataclass
class PredictionResult:
    """预测结果数据类"""
    label: str
    label_cn: str
    confidence: float
    process_time_ms: float
    model_version: str = "v1.0"
    timestamp: float = None
    all_confidences: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class ErrorResponse:
    """错误响应数据类"""
    error: str
    error_type: str
    timestamp: float = None
    details: Optional[str] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

# =================异常处理=================
class ModelException(Exception): 
    """模型相关异常"""
    pass

class ValidationException(Exception): 
    """数据验证异常"""
    pass

class ServiceException(Exception): 
    """服务相关异常"""
    pass

class OllamaException(Exception): 
    """Ollama服务异常"""
    pass

def handle_exceptions(f):
    """统一异常处理装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = f"{int(time.time() * 1000)}-{id(request)}"
        try:
            logger.info(f"[{request_id}] 请求开始: {request.method} {request.path}")
            result = f(*args, **kwargs)
            logger.info(f"[{request_id}] 请求成功")
            return result
            
        except ValidationException as e:
            logger.warning(f"[{request_id}] 验证异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="ValidationError",
                request_id=request_id
            ).to_dict()), 400
            
        except ModelException as e:
            logger.error(f"[{request_id}] 模型异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="ModelError",
                request_id=request_id
            ).to_dict()), 500
            
        except OllamaException as e:
            logger.error(f"[{request_id}] Ollama异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="OllamaError",
                request_id=request_id
            ).to_dict()), 503
            
        except ServiceException as e:
            logger.error(f"[{request_id}] 服务异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="ServiceError",
                request_id=request_id
            ).to_dict()), 503
            
        except Exception as e:
            logger.error(f"[{request_id}] 未知异常: {str(e)}\n{traceback.format_exc()}")
            return jsonify(ErrorResponse(
                error="Internal server error", 
                error_type="UnknownError",
                details=str(e) if config.DEBUG else None,
                request_id=request_id
            ).to_dict()), 500
            
    return decorated_function

# =================模型定义=================
class MedicalBertClassifier(nn.Module):
    """医疗文本分类模型"""
    
    def __init__(self, bert_model_name: str, num_classes: int = 3, dropout: float = 0.3):
        super(MedicalBertClassifier, self).__init__()
        logger.info(f"初始化BERT模型: {bert_model_name}")
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
        logger.info("模型结构初始化完成")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        hidden = self.fc1(pooled_output)
        hidden = self.relu(hidden)
        hidden = self.dropout2(hidden)
        logits = self.classifier(hidden)
        return logits

# =================模型服务=================
class ModelService:
    """模型服务单例类"""
    _instance: Optional['ModelService'] = None
    _initialized: bool = False
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized: 
            return
            
        logger.info("="*60)
        logger.info("初始化模型服务")
        logger.info("="*60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU信息: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self.model = None
        self.tokenizer = None
        self.label_map = {0: "abnormal", 1: "normal", 2: "review"}
        self.model_version = "v1.0"
        self.load_time = None
        self.predict_count = 0
        
        self._load_resources()
        self._initialized = True
        logger.info("模型服务初始化完成")
    
    def _load_resources(self) -> None:
        """加载模型资源"""
        start_time = time.time()
        
        try:
            # 加载tokenizer
            logger.info(f"加载Tokenizer: {config.BERT_MODEL_NAME}")
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
            logger.info(f"✓ Tokenizer加载成功, 词表大小: {len(self.tokenizer)}")
            
            # 初始化模型
            logger.info("初始化模型结构")
            self.model = MedicalBertClassifier(config.BERT_MODEL_NAME, num_classes=3)
            
            # 加载模型权重
            if os.path.exists(config.MODEL_PATH):
                logger.info(f"加载模型权重: {config.MODEL_PATH}")
                state_dict = torch.load(config.MODEL_PATH, map_location=self.device)
                
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    if 'epoch' in state_dict:
                        logger.info(f"模型训练轮次: {state_dict['epoch']}")
                    if 'best_acc' in state_dict:
                        logger.info(f"最佳准确率: {state_dict['best_acc']:.4f}")
                else:
                    self.model.load_state_dict(state_dict)
                
                logger.info("✓ 模型权重加载成功")
            else:
                logger.warning(f"!!! 模型文件不存在: {config.MODEL_PATH}")
                logger.warning("将使用随机初始化权重（仅供测试）")
            
            # 移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            self.load_time = time.time() - start_time
            logger.info(f"✓ 模型加载完成! 耗时: {self.load_time:.2f}秒")
            
            # 预热推理
            self._warmup()
            
        except Exception as e:
            logger.error(f"✗ 模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelException(f"模型加载失败: {str(e)}")
    
    def _warmup(self) -> None:
        """模型预热"""
        try:
            logger.info("执行模型预热...")
            test_text = "血红蛋白: 150 g/L, 参考范围: 120-160"
            self.predict(test_text)
            logger.info("✓ 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {str(e)}")
    
    def _validate_input(self, text: str) -> None:
        """验证输入数据"""
        if not text:
            raise ValidationException("输入文本为空")
        
        if not isinstance(text, str):
            raise ValidationException(f"输入类型错误，期望str，实际{type(text)}")
        
        text = text.strip()
        if not text:
            raise ValidationException("输入文本仅包含空白字符")
        
        if len(text) > 1000:
            raise ValidationException(f"输入文本过长: {len(text)}字符（最大1000）")
    
    def predict(self, text: str) -> PredictionResult:
        """执行预测"""
        start_time = time.time()
        
        try:
            # 验证输入
            self._validate_input(text)
            text = text.strip()
            
            logger.debug(f"预测文本: {text[:100]}...")
            
            # 编码
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=config.MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                confidence_scores = probs[0].cpu().numpy()
                _, pred = torch.max(outputs, dim=1)
            
            # 解析结果
            pred_idx = pred.item()
            pred_label_str = self.label_map[pred_idx]
            pred_label = PredictionLabel(pred_label_str)
            confidence = float(confidence_scores[pred_idx])
            
            # 所有类别的置信度
            all_confidences = {
                self.label_map[i]: float(confidence_scores[i]) 
                for i in range(len(confidence_scores))
            }
            
            # 更新统计
            self.predict_count += 1
            process_time = (time.time() - start_time) * 1000
            
            logger.debug(f"预测完成: {pred_label.value} ({confidence:.4f}), 耗时: {process_time:.2f}ms")
            
            result = PredictionResult(
                label=pred_label.value,
                label_cn=pred_label.chinese,
                confidence=round(confidence, 4),
                process_time_ms=round(process_time, 2),
                model_version=self.model_version,
                all_confidences=all_confidences
            )
            
            return result
            
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelException(f"预测过程出错: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Ollama
            ollama_status = "unknown"
            ollama_models = []
            
            try:
                response = requests.get(
                    f"{config.OLLAMA_HOST}/api/tags", 
                    timeout=5
                )
                if response.status_code == 200:
                    ollama_status = "healthy"
                    data = response.json()
                    ollama_models = [m.get('name', '') for m in data.get('models', [])]
                else:
                    ollama_status = f"unhealthy (status: {response.status_code})"
            except requests.exceptions.ConnectionError:
                ollama_status = "unreachable"
            except Exception as e:
                ollama_status = f"error: {str(e)}"
            
            return {
                "status": "healthy" if self.model else "unhealthy",
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None,
                "load_time_seconds": round(self.load_time, 2) if self.load_time else None,
                "predict_count": self.predict_count,
                "ollama_status": ollama_status,
                "ollama_host": config.OLLAMA_HOST,
                "ollama_models": ollama_models,
                "timestamp": time.time(),
                "uptime_seconds": round(time.time() - (time.time() - self.load_time if self.load_time else 0), 2)
            }
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model": "MedicalBertClassifier",
            "version": self.model_version,
            "bert_base": config.BERT_MODEL_NAME,
            "device": str(self.device),
            "labels": list(self.label_map.values()),
            "label_mapping": {v: PredictionLabel(v).chinese for v in self.label_map.values()},
            "max_text_length": config.MAX_TEXT_LENGTH,
            "predict_count": self.predict_count
        }

# =================Ollama代理服务=================
class OllamaProxy:
    """Ollama服务代理"""
    
    def __init__(self):
        self.request_count = 0
        logger.info(f"初始化Ollama代理: {config.OLLAMA_HOST}")
    
    def _retry_request(self, func, max_retries: int = config.MAX_RETRIES):
        """带重试的请求"""
        for attempt in range(max_retries):
            try:
                return func()
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama请求超时，重试 {attempt + 1}/{max_retries}")
                    time.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    raise
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama连接失败，重试 {attempt + 1}/{max_retries}")
                    time.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    raise
    
    def generate(self, prompt: str, model: str = "qwen2.5", stream: bool = False, **kwargs) -> Any:
        """生成文本"""
        try:
            url = f"{config.OLLAMA_HOST}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            
            logger.info(f"Ollama请求: model={model}, stream={stream}, prompt_len={len(prompt)}")
            start_time = time.time()
            
            def make_request():
                return requests.post(
                    url,
                    json=payload,
                    timeout=config.OLLAMA_TIMEOUT,
                    stream=stream
                )
            
            response = self._retry_request(make_request)
            response.raise_for_status()
            
            self.request_count += 1
            
            if stream:
                # 流式响应
                def generate_stream() -> Generator[bytes, None, None]:
                    try:
                        total_bytes = 0
                        for chunk in response.iter_content(chunk_size=config.OLLAMA_STREAM_CHUNK_SIZE):
                            if chunk:
                                total_bytes += len(chunk)
                                yield chunk
                        
                        elapsed = time.time() - start_time
                        logger.info(f"Ollama流式响应完成: {total_bytes}字节, 耗时: {elapsed:.2f}秒")
                    except Exception as e:
                        logger.error(f"流式响应错误: {str(e)}")
                        raise
                
                return generate_stream()
            else:
                # 非流式响应
                result = response.json()
                elapsed = time.time() - start_time
                response_text = result.get('response', '')
                logger.info(f"Ollama响应完成: {len(response_text)}字符, 耗时: {elapsed:.2f}秒")
                return result
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama请求超时 (>{config.OLLAMA_TIMEOUT}秒)")
            raise OllamaException(f"Ollama服务响应超时（>{config.OLLAMA_TIMEOUT}秒）")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama连接失败: {str(e)}")
            raise OllamaException(f"无法连接到Ollama服务 ({config.OLLAMA_HOST})")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama请求失败: {str(e)}")
            raise OllamaException(f"Ollama服务请求失败: {str(e)}")
    
    def get_models(self) -> Dict[str, Any]:
        """获取可用模型"""
        try:
            def make_request():
                return requests.get(
                    f"{config.OLLAMA_HOST}/api/tags", 
                    timeout=10
                )
            
            response = self._retry_request(make_request)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"获取到 {len(data.get('models', []))} 个Ollama模型")
            return data
            
        except Exception as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            return {"models": [], "error": str(e)}

# =================Flask应用=================
class FlaskApp:
    """Flask应用封装"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = Flask(__name__)
        self.model_service = None
        self.ollama_proxy = OllamaProxy()
        self.start_time = time.time()
        
        self._setup_app()
        self._setup_cors()
        self._register_routes()
        self._setup_error_handlers()
        
        logger.info("Flask应用初始化完成")
    
    def _setup_app(self):
        """配置Flask应用"""
        self.app.config['JSON_AS_ASCII'] = False
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
        WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    def _setup_cors(self):
        """配置CORS"""
        CORS(
            self.app,
            resources={r"/*": {
                "origins": self.config.CORS_ORIGINS,
                "methods": self.config.CORS_METHODS,
                "allow_headers": self.config.CORS_HEADERS,
                "expose_headers": ["Content-Type", "X-Request-ID"],
                "supports_credentials": True,
                "max_age": 3600
            }}
        )
        logger.info("CORS配置完成")
    
    def _setup_error_handlers(self):
        """设置错误处理器"""
        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({"error": "接口不存在", "error_type": "NotFound"}), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(e):
            return jsonify({"error": "不支持的HTTP方法", "error_type": "MethodNotAllowed"}), 405
        
        @self.app.errorhandler(413)
        def request_entity_too_large(e):
            return jsonify({"error": "请求体过大", "error_type": "RequestTooLarge"}), 413
    
    def _register_routes(self):
        """注册路由"""
        
        @self.app.route('/')
        def home():
            """首页"""
            uptime = time.time() - self.start_time
            return jsonify({
                "status": "running",
                "service": "Medical AI Backend",
                "version": "1.0.0",
                "uptime_seconds": round(uptime, 2),
                "endpoints": {
                    "predict": {
                        "path": "/api/predict",
                        "method": "POST",
                        "description": "单条文本分析"
                    },
                    "batch_predict": {
                        "path": "/api/batch_predict",
                        "method": "POST",
                        "description": "批量文本分析"
                    },
                    "health": {
                        "path": "/api/health",
                        "method": "GET",
                        "description": "健康检查"
                    },
                    "info": {
                        "path": "/api/info",
                        "method": "GET",
                        "description": "模型信息"
                    },
                    "ollama_generate": {
                        "path": "/api/ollama/generate",
                        "method": "POST",
                        "description": "Ollama文本生成"
                    },
                    "ollama_models": {
                        "path": "/api/ollama/models",
                        "method": "GET",
                        "description": "Ollama可用模型"
                    }
                },
                "api_base_url": f"http://{request.host}",
                "config": {
                    "bert_model": config.BERT_MODEL_NAME,
                    "ollama_host": config.OLLAMA_HOST,
                    "max_text_length": config.MAX_TEXT_LENGTH,
                    "batch_max_size": config.BATCH_MAX_SIZE
                }
            })
        
        @self.app.route('/api/predict', methods=['POST'])
        @handle_exceptions
        def predict():
            """单条预测"""
            data = request.get_json(silent=True)
            
            if not data:
                raise ValidationException("请求体为空或格式错误")
            
            if 'text' not in data:
                raise ValidationException("缺少必需字段: text")
            
            text = data['text']
            result = self.model_service.predict(text)
            
            return jsonify(result.to_dict())
        
        @self.app.route('/api/batch_predict', methods=['POST'])
        @handle_exceptions
        def batch_predict():
            """批量预测"""
            data = request.get_json(silent=True)
            
            if not data:
                raise ValidationException("请求体为空或格式错误")
            
            if 'texts' not in data:
                raise ValidationException("缺少必需字段: texts")
            
            texts = data['texts']
            
            if not isinstance(texts, list):
                raise ValidationException("texts字段必须是数组")
            
            if len(texts) == 0:
                raise ValidationException("texts数组不能为空")
            
            if len(texts) > config.BATCH_MAX_SIZE:
                raise ValidationException(f"批量大小超限: {len(texts)} (最大{config.BATCH_MAX_SIZE})")
            
            logger.info(f"批量预测开始: {len(texts)}条")
            start_time = time.time()
            
            results = []
            successful = 0
            
            for idx, text in enumerate(texts):
                try:
                    result = self.model_service.predict(text)
                    results.append(result.to_dict())
                    successful += 1
                except Exception as e:
                    logger.warning(f"批量预测第{idx+1}条失败: {str(e)}")
                    results.append({
                        "error": str(e),
                        "text_preview": text[:50] + "..." if len(text) > 50 else text,
                        "index": idx
                    })
            
            elapsed = time.time() - start_time
            logger.info(f"批量预测完成: {successful}/{len(texts)}, 耗时: {elapsed:.2f}秒")
            
            return jsonify({
                "results": results,
                "total": len(texts),
                "successful": successful,
                "failed": len(texts) - successful,
                "process_time_seconds": round(elapsed, 2)
            })
        
        @self.app.route('/api/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify(self.model_service.health_check())
        
        @self.app.route('/api/info', methods=['GET'])
        def info():
            """模型信息"""
            return jsonify(self.model_service.get_model_info())
        
        @self.app.route('/api/ollama/generate', methods=['POST'])
        @handle_exceptions
        def ollama_generate():
            """Ollama生成"""
            data = request.get_json(silent=True)
            
            if not data:
                raise ValidationException("请求体为空")
            
            prompt = data.get('prompt', '')
            if not prompt:
                raise ValidationException("缺少必需字段: prompt")
            
            model = data.get('model', 'qwen2.5')
            stream = data.get('stream', False)
            options = data.get('options', {})
            
            logger.info(f"Ollama生成请求: model={model}, stream={stream}")
            
            result = self.ollama_proxy.generate(
                prompt=prompt,
                model=model,
                stream=stream,
                **options
            )
            
            if stream:
                # 流式响应
                return Response(result, mimetype='application/json')
            else:
                # 普通响应
                return jsonify(result)
        
        @self.app.route('/api/ollama/models', methods=['GET'])
        def ollama_models():
            """Ollama模型列表"""
            models = self.ollama_proxy.get_models()
            return jsonify(models)
        
        @self.app.route('/api/stats', methods=['GET'])
        def stats():
            """服务统计"""
            uptime = time.time() - self.start_time
            return jsonify({
                "uptime_seconds": round(uptime, 2),
                "model_predict_count": self.model_service.predict_count,
                "ollama_request_count": self.ollama_proxy.request_count,
                "timestamp": time.time()
            })
    
    def initialize_model(self):
        """初始化模型服务"""
        logger.info("开始初始化模型服务...")
        self.model_service = ModelService()
        logger.info("模型服务初始化成功")
    
    def run(self):
        """运行服务"""
        logger.info("="*60)
        logger.info("医疗AI后端服务启动")
        logger.info("="*60)
        logger.info(f"监听地址: {self.config.HOST}:{self.config.PORT}")
        logger.info(f"本地访问: http://127.0.0.1:{self.config.PORT}")
        logger.info(f"外部访问: http://服务器IP:{self.config.PORT}")
        logger.info(f"Ollama服务: {config.OLLAMA_HOST}")
        logger.info(f"模型路径: {config.MODEL_PATH}")
        logger.info(f"调试模式: {config.DEBUG}")
        logger.info("="*60)
        
        self.app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG,
            threaded=True,
            use_reloader=False
        )

# =================主程序入口=================
def check_dependencies():
    """检查依赖"""
    missing = []
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import flask
    except ImportError:
        missing.append("flask")
    
    try:
        import flask_cors
    except ImportError:
        missing.append("flask-cors")
    
    if missing:
        logger.error(f"缺少依赖包: {', '.join(missing)}")
        logger.error(f"请运行: pip install {' '.join(missing)}")
        return False
    
    return True

if __name__ == '__main__':
    try:
        # 检查依赖
        if not check_dependencies():
            sys.exit(1)
        
        # 创建并运行应用
        flask_app = FlaskApp(config)
        flask_app.initialize_model()
        flask_app.run()
        
    except KeyboardInterrupt:
        logger.info("收到终止信号，服务关闭")
    except Exception as e:
        logger.critical(f"启动失败: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)