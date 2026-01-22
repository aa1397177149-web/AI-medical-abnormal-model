import os
import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
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
import re

# =================配置区域=================
@dataclass
class Config:
    """统一配置管理"""
    # 模型配置
    MODEL_PATH: str = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'best_medical_model.pth'))
    BERT_MODEL_NAME: str = os.environ.get('BERT_MODEL_NAME', 'trueto/medbert-base-wwm-chinese')

    # 服务器配置
    PORT: int = int(os.environ.get('PORT', 6006))
    HOST: str = os.environ.get('HOST', '0.0.0.0')
    DEBUG: bool = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # 日志配置
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE: str = 'logs/server.log'
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # 模型配置
    MAX_TEXT_LENGTH: int = 128
    REQUEST_TIMEOUT: int = 30
    BATCH_MAX_SIZE: int = 100
    MIN_TEXT_LENGTH: int = 5
    MAX_INPUT_SIZE: int = 1000
    
    # Ollama配置
    OLLAMA_HOST: str = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_TIMEOUT: int = 180
    OLLAMA_STREAM_CHUNK_SIZE: int = 1024
    OLLAMA_MAX_PROMPT_LENGTH: int = 8000
    
    # CORS配置
    CORS_ORIGINS: str = "*"
    CORS_METHODS: list = None
    CORS_HEADERS: list = None
    
    # 重试配置
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # 速率限制
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
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

# =================数据验证=================
class InputValidator:
    """输入数据验证器"""
    
    @staticmethod
    def validate_text(text: str, field_name: str = "text") -> str:
        """验证文本输入"""
        if not text:
            raise ValidationException(f"{field_name}不能为空")
        
        if not isinstance(text, str):
            raise ValidationException(f"{field_name}必须是字符串类型")
        
        text = text.strip()
        if not text:
            raise ValidationException(f"{field_name}不能仅包含空白字符")
        
        if len(text) < config.MIN_TEXT_LENGTH:
            raise ValidationException(
                f"{field_name}太短: {len(text)}字符（最少{config.MIN_TEXT_LENGTH}字符）"
            )
        
        if len(text) > config.MAX_INPUT_SIZE:
            raise ValidationException(
                f"{field_name}太长: {len(text)}字符（最多{config.MAX_INPUT_SIZE}字符）"
            )
        
        # 检查是否包含有效内容
        if not re.search(r'[\w\u4e00-\u9fff]', text):
            raise ValidationException(f"{field_name}不包含有效字符")
        
        return text
    
    @staticmethod
    def validate_batch_texts(texts: list) -> list:
        """验证批量文本输入"""
        if not isinstance(texts, list):
            raise ValidationException("texts字段必须是数组")
        
        if len(texts) == 0:
            raise ValidationException("texts数组不能为空")
        
        if len(texts) > config.BATCH_MAX_SIZE:
            raise ValidationException(
                f"批量大小超限: {len(texts)} (最大{config.BATCH_MAX_SIZE})"
            )
        
        validated_texts = []
        for idx, text in enumerate(texts):
            try:
                validated = InputValidator.validate_text(text, f"texts[{idx}]")
                validated_texts.append(validated)
            except ValidationException as e:
                logger.warning(f"批量验证第{idx+1}项失败: {str(e)}")
                raise ValidationException(f"第{idx+1}项验证失败: {str(e)}")
        
        return validated_texts
    
    @staticmethod
    def validate_ollama_params(prompt: str, model: str, **kwargs) -> tuple:
        """验证Ollama参数"""
        # 验证prompt
        prompt = InputValidator.validate_text(prompt, "prompt")
        
        if len(prompt) > config.OLLAMA_MAX_PROMPT_LENGTH:
            raise ValidationException(
                f"prompt太长: {len(prompt)}字符 (最大{config.OLLAMA_MAX_PROMPT_LENGTH})"
            )
        
        # 验证model
        if not model or not isinstance(model, str):
            raise ValidationException("model参数无效")
        
        if not re.match(r'^[a-zA-Z0-9._:-]+$', model):
            raise ValidationException("model名称包含非法字符")
        
        # 验证options
        if 'options' in kwargs:
            options = kwargs['options']
            if not isinstance(options, dict):
                raise ValidationException("options必须是字典类型")
            
            # 验证temperature
            if 'temperature' in options:
                temp = options['temperature']
                if not isinstance(temp, (int, float)):
                    raise ValidationException("temperature必须是数字")
                if temp < 0 or temp > 2:
                    raise ValidationException("temperature必须在0-2之间")
        
        return prompt, model

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
    suggestions: Optional[list] = None
    
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

class RateLimitException(Exception):
    """速率限制异常"""
    pass

def handle_exceptions(f):
    """统一异常处理装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = f"{int(time.time() * 1000)}-{id(request)}"
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] 请求开始: {request.method} {request.path}")
            result = f(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] 请求成功 ({elapsed:.2f}ms)")
            return result
            
        except ValidationException as e:
            logger.warning(f"[{request_id}] 验证异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="ValidationError",
                request_id=request_id,
                suggestions=["请检查输入数据格式", "确保所有必需字段都已填写"]
            ).to_dict()), 400
            
        except ModelException as e:
            logger.error(f"[{request_id}] 模型异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="ModelError",
                request_id=request_id,
                suggestions=["请稍后重试", "如问题持续，请联系管理员"]
            ).to_dict()), 500
            
        except OllamaException as e:
            logger.error(f"[{request_id}] Ollama异常: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e), 
                error_type="OllamaError",
                request_id=request_id,
                suggestions=["检查Ollama服务是否运行", "验证模型是否已安装"]
            ).to_dict()), 503
            
        except RateLimitException as e:
            logger.warning(f"[{request_id}] 速率限制: {str(e)}")
            return jsonify(ErrorResponse(
                error=str(e),
                error_type="RateLimitError",
                request_id=request_id,
                suggestions=["请稍后再试", "降低请求频率"]
            ).to_dict()), 429
            
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

# =================速率限制=================
class RateLimiter:
    """简单的速率限制器"""
    
    def __init__(self):
        self.requests = {}
        self.lock = threading.Lock()
    
    def check_rate_limit(self, client_id: str) -> bool:
        """检查是否超过速率限制"""
        if not config.RATE_LIMIT_ENABLED:
            return True
        
        current_time = time.time()
        
        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # 清理过期记录
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if current_time - t < config.RATE_LIMIT_WINDOW
            ]
            
            # 检查限制
            if len(self.requests[client_id]) >= config.RATE_LIMIT_REQUESTS:
                return False
            
            # 记录请求
            self.requests[client_id].append(current_time)
            return True
    
    def get_client_id(self, request) -> str:
        """获取客户端标识"""
        # 使用IP地址作为标识
        return request.remote_addr or 'unknown'

rate_limiter = RateLimiter()

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
        self.error_count = 0
        
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
    
    def predict(self, text: str) -> PredictionResult:
        """执行预测"""
        start_time = time.time()
        
        try:
            # 验证输入
            text = InputValidator.validate_text(text)
            
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
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
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
                "error_count": self.error_count,
                "success_rate": round(
                    (self.predict_count - self.error_count) / max(self.predict_count, 1) * 100, 
                    2
                ) if self.predict_count > 0 else 100,
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
            "predict_count": self.predict_count,
            "error_count": self.error_count
        }

# =================Ollama代理服务=================
class OllamaProxy:
    """Ollama服务代理"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        logger.info(f"初始化Ollama代理: {config.OLLAMA_HOST}")
    
    def _retry_request(self, func, max_retries: int = config.MAX_RETRIES):
        """带重试的请求"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama请求超时，重试 {attempt + 1}/{max_retries}")
                    time.sleep(config.RETRY_DELAY * (attempt + 1))
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama连接失败，重试 {attempt + 1}/{max_retries}")
                    time.sleep(config.RETRY_DELAY * (attempt + 1))
        
        # 所有重试都失败
        self.error_count += 1
        raise last_exception
    
    def generate(self, prompt: str, model: str = "qwen2.5:3b", stream: bool = False, **kwargs) -> Any:
        """生成文本"""
        try:
            # 验证参数
            prompt, model = InputValidator.validate_ollama_params(prompt, model, **kwargs)
            
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
                        self.error_count += 1
                        raise
                
                return generate_stream()
            else:
                # 非流式响应
                result = response.json()
                elapsed = time.time() - start_time
                response_text = result.get('response', '')
                logger.info(f"Ollama响应完成: {len(response_text)}字符, 耗时: {elapsed:.2f}秒")
                return result
                
        except ValidationException:
            self.error_count += 1
            raise
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama请求超时 (>{config.OLLAMA_TIMEOUT}秒)")
            raise OllamaException(f"Ollama服务响应超时（>{config.OLLAMA_TIMEOUT}秒）")
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama连接失败: {str(e)}")
            raise OllamaException(f"无法连接到Ollama服务: {config.OLLAMA_HOST}")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Ollama连接失败: {str(e)}")
            raise OllamaException(f"无法连接到Ollama服务: {config.OLLAMA_HOST}")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Ollama生成失败: {str(e)}")
            raise OllamaException(f"生成过程出错: {str(e)}")

    def list_models(self) -> dict:
        """获取模型列表"""
        try:
            url = f"{config.OLLAMA_HOST}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"models": []}
        except Exception as e:
            logger.warning(f"获取Ollama模型列表失败: {e}")
            return {"models": []}

# 初始化服务实例
model_service = ModelService()
ollama_proxy = OllamaProxy()

# =================规则引擎=================
class RuleEngine:
    """
    增强的规则引擎
    支持多种参考值格式：数值范围、不等式、定性描述等
    """
    @staticmethod
    def extract_result_and_reference(text: str) -> Tuple[Optional[str], Optional[str]]:
        """从文本中提取结果和参考值（字符串形式）"""
        try:
            # 提取结果
            res_match = re.search(r'结果[:：]\s*([^,，]+)', text)
            if not res_match:
                return None, None
            result = res_match.group(1).strip()

            # 提取参考范围/参考值
            ref_match = re.search(r'参考(?:范围)?[:：]\s*([^,，]+)', text)
            if not ref_match:
                return result, None
            reference = ref_match.group(1).strip()

            return result, reference
        except Exception as e:
            logger.debug(f"提取结果和参考值失败: {e}")
            return None, None

    @staticmethod
    def judge_numeric_range(result: str, reference: str) -> Optional[bool]:
        """判断数值范围类型的参考值"""
        try:
            # 处理 "0--1" 或 "3.5-9.5" 格式
            range_match = re.search(r'(\d+\.?\d*)\s*-+\s*(\d+\.?\d*)', reference)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))

                # 尝试提取结果中的数值
                result_num_match = re.search(r'(\d+\.?\d*)', result)
                if result_num_match:
                    result_val = float(result_num_match.group(1))
                    return low <= result_val <= high

            # 处理 "≤3" 或 "<3" 或 "<=3" 格式
            le_match = re.search(r'[≤<=]\s*(\d+\.?\d*)', reference)
            if le_match:
                max_val = float(le_match.group(1))
                result_num_match = re.search(r'(\d+\.?\d*)', result)
                if result_num_match:
                    result_val = float(result_num_match.group(1))
                    return result_val <= max_val

            # 处理 "≥3" 或 ">3" 或 ">=3" 格式
            ge_match = re.search(r'[≥>=]\s*(\d+\.?\d*)', reference)
            if ge_match:
                min_val = float(ge_match.group(1))
                result_num_match = re.search(r'(\d+\.?\d*)', result)
                if result_num_match:
                    result_val = float(result_num_match.group(1))
                    return result_val >= min_val

            return None
        except Exception as e:
            logger.debug(f"数值范围判断失败: {e}")
            return None

    @staticmethod
    def judge_qualitative(result: str, reference: str) -> Optional[bool]:
        """判断定性描述类型的参考值"""
        try:
            # 标准化文本（去除空格、括号等）
            result_clean = re.sub(r'[\s\(\)\[\]（）【】]', '', result.lower())
            reference_clean = re.sub(r'[\s\(\)\[\]（）【】]', '', reference.lower())

            # 阴性相关
            negative_keywords = ['阴性', '阴', '-', 'negative', 'neg']
            positive_keywords = ['阳性', '阳', '+', 'positive', 'pos']

            # 正常相关
            normal_keywords = ['正常', '清澈', '清晰', '未见异常', '无异常', '正常范围', 'normal', '清']
            abnormal_keywords = ['异常', '阳性', '浑浊', '混浊', '增高', '降低', 'abnormal']

            # 判断参考值是否要求阴性
            ref_requires_negative = any(kw in reference_clean for kw in negative_keywords)
            if ref_requires_negative:
                # 检查结果是否为阴性
                result_is_negative = any(kw in result_clean for kw in negative_keywords)
                result_is_positive = any(kw in result_clean for kw in positive_keywords)
                if result_is_negative:
                    return True  # 正常
                if result_is_positive:
                    return False  # 异常

            # 判断参考值是否要求正常/清澈等
            ref_requires_normal = any(kw in reference_clean for kw in normal_keywords)
            if ref_requires_normal:
                # 检查结果是否正常
                result_is_normal = any(kw in result_clean for kw in normal_keywords)
                result_is_abnormal = any(kw in result_clean for kw in abnormal_keywords)
                if result_is_normal:
                    return True  # 正常
                if result_is_abnormal:
                    return False  # 异常

            # 完全匹配
            if result_clean == reference_clean:
                return True

            return None
        except Exception as e:
            logger.debug(f"定性判断失败: {e}")
            return None

    @staticmethod
    def judge(text: str) -> Optional[PredictionResult]:
        """
        尝试使用规则判断
        返回: 如果规则匹配成功返回PredictionResult，否则返回None
        """
        try:
            result, reference = RuleEngine.extract_result_and_reference(text)

            # 如果没有提取到结果或参考值，返回None，交给BERT处理
            if result is None or reference is None:
                return None

            logger.debug(f"规则引擎: 结果='{result}', 参考='{reference}'")

            # 尝试数值范围判断
            is_normal = RuleEngine.judge_numeric_range(result, reference)

            # 如果数值判断失败，尝试定性判断
            if is_normal is None:
                is_normal = RuleEngine.judge_qualitative(result, reference)

            # 如果仍然无法判断，返回None交给BERT
            if is_normal is None:
                logger.debug(f"规则引擎无法判断，交给BERT: {text}")
                return None

            # 确定标签
            label = PredictionLabel.NORMAL if is_normal else PredictionLabel.ABNORMAL

            logger.info(f"规则引擎判断: {result} vs {reference} -> {label.chinese}")

            # 构造返回结果，置信度设为1.0
            return PredictionResult(
                label=label.value,
                label_cn=label.chinese,
                confidence=1.0,
                process_time_ms=0.0,
                model_version="RuleBased-v2.0",
                all_confidences={"normal": 1.0 if label == PredictionLabel.NORMAL else 0.0,
                                 "abnormal": 1.0 if label == PredictionLabel.ABNORMAL else 0.0,
                                 "review": 0.0}
            )
        except Exception as e:
            logger.debug(f"规则判断异常: {e}")
            return None

# =================Flask 应用=================
app = Flask(__name__)
# 配置CORS
CORS(app, resources={
    r"/api/*": {
        "origins": config.CORS_ORIGINS,
        "methods": config.CORS_METHODS,
        "allow_headers": config.CORS_HEADERS
    }
})

@app.before_request
def before_request():
    """请求前置处理"""
    # 速率限制检查
    client_id = rate_limiter.get_client_id(request)
    if not rate_limiter.check_rate_limit(client_id):
        raise RateLimitException("请求过于频繁，请稍后再试")

@app.route('/api/health', methods=['GET'])
@handle_exceptions
def health_check():
    """服务健康检查"""
    return jsonify(model_service.health_check())

@app.route('/api/info', methods=['GET'])
@handle_exceptions
def model_info():
    """获取模型详情"""
    return jsonify(model_service.get_model_info())

@app.route('/api/predict', methods=['POST'])
@handle_exceptions
def predict():
    """
    单条预测接口
    逻辑：优先使用规则引擎 -> 规则无法处理则使用BERT模型
    """
    data = request.get_json()
    if not data:
        raise ValidationException("请求体不能为空")
        
    text = data.get('text')
    text = InputValidator.validate_text(text)
    
    # 1. 尝试规则判断 (处理有明确参考数值的项目)
    rule_result = RuleEngine.judge(text)
    if rule_result:
        logger.info(f"规则命中: {text[:30]}... -> {rule_result.label}")
        return jsonify(rule_result.to_dict())
    
    # 2. BERT模型预测 (处理无参考值、定性描述或文本复杂的项目)
    result = model_service.predict(text)
    return jsonify(result.to_dict())

@app.route('/api/batch_predict', methods=['POST'])
@handle_exceptions
def batch_predict():
    """批量预测接口"""
    data = request.get_json()
    if not data:
        raise ValidationException("请求体不能为空")
        
    texts = data.get('texts', [])
    texts = InputValidator.validate_batch_texts(texts)
    
    results = []
    for text in texts:
        try:
            # 同样采用 规则 -> BERT 的混合逻辑
            rule_result = RuleEngine.judge(text)
            if rule_result:
                results.append(rule_result.to_dict())
            else:
                res = model_service.predict(text)
                results.append(res.to_dict())
        except Exception as e:
            # 批量处理中某一条失败不应中断整个请求
            logger.error(f"批量处理单条失败: {e}")
            results.append({
                "label": "unknown",
                "label_cn": "未知",
                "confidence": 0.0,
                "error": str(e)
            })
            
    return jsonify({"results": results, "count": len(results)})

@app.route('/api/ollama/generate', methods=['POST'])
@handle_exceptions
def ollama_generate():
    """
    Ollama 生成接口 (用于生成单个病人的综述)
    前端已将单个病人的所有异常项和基本信息整理成 Prompt 发送至此
    """
    data = request.get_json()
    if not data:
        raise ValidationException("请求体不能为空")
        
    prompt = data.get('prompt')
    model = data.get('model', 'qwen2.5:3b')
    stream = data.get('stream', False)
    options = data.get('options', {})
    
    # 代理调用 Ollama
    result = ollama_proxy.generate(
        prompt=prompt,
        model=model,
        stream=stream,
        options=options
    )
    
    if stream:
        # 如果是流式响应，直接返回生成器
        return Response(result, mimetype='application/x-ndjson')
    
    return jsonify(result)

@app.route('/api/ollama/models', methods=['GET'])
@handle_exceptions
def ollama_models():
    """获取可用的 Ollama 模型"""
    return jsonify(ollama_proxy.list_models())

@app.route('/')
def index():
    """主页路由 - 渲染前端页面"""
    return send_from_directory('templates', 'index.html')

# =================启动入口=================
if __name__ == '__main__':
    try:
        # 打印启动 Banner
        print(f"""
        ========================================
        医疗报告 AI 智能工作台 Backend Service
        ========================================
        Server: {config.HOST}:{config.PORT}
        Device: {model_service.device}
        BERT Model: {config.BERT_MODEL_NAME}
        Ollama Host: {config.OLLAMA_HOST}
        Debug Mode: {config.DEBUG}
        ========================================
        """)
        
        # 启动 Flask 服务
        # 注意: 在生产环境中应使用 Gunicorn 或 uWSGI
        app.run(
            host=config.HOST, 
            port=config.PORT, 
            debug=config.DEBUG,
            threaded=True,
            use_reloader=False  # 防止加载两次模型
        )
    except Exception as e:
        logger.critical(f"服务启动失败: {e}")
        sys.exit(1)