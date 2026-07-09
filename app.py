#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病害材料预测API服务
提供RESTful API接口供外部调用，支持Swagger文档、API Key认证和增强日志
"""

import os
import json
import logging
import time
import uuid
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

API_TITLE = os.getenv('API_TITLE', '病害材料预测API')
API_VERSION = os.getenv('API_VERSION', '1.0')
API_DESCRIPTION = os.getenv('API_DESCRIPTION', '道路病害材料用量预测服务，基于案例匹配算法进行智能材料清单预测')

api = Api(
    app,
    version=API_VERSION,
    title=API_TITLE,
    description=API_DESCRIPTION,
    doc='/swagger/',
    prefix='/api',
    security='apiKey',
    authorizations={
        'apiKey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-Key',
            'description': 'API Key认证，在请求头中添加 X-API-Key: your_api_key'
        }
    }
)

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('LOG_FILE', 'app.log')


class ContextFilter(logging.Filter):
    """为日志记录注入请求上下文默认值，避免 KeyError"""

    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = '-'
        if not hasattr(record, 'remote_addr'):
            record.remote_addr = '-'
        if not hasattr(record, 'endpoint'):
            record.endpoint = '-'
        return True


_context_filter = ContextFilter()

_file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
_file_handler.addFilter(_context_filter)
_stream_handler = logging.StreamHandler()
_stream_handler.addFilter(_context_filter)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(request_id)s - %(remote_addr)s - %(endpoint)s - %(message)s',
    handlers=[_file_handler, _stream_handler]
)

logger = logging.getLogger(__name__)

VALID_API_KEYS = set(filter(None, os.getenv('API_KEYS', '').split(',')))
AUTH_REQUIRED = os.getenv('AUTH_REQUIRED', 'true').lower() == 'true'

ns = api.namespace('predictor', description='病害材料预测相关接口')

predict_request_model = api.model('PredictRequest', {
    'disease_type': fields.String(
        required=True,
        description='病害类型，如：车辙、坑槽、坑洞、拱起、接缝、横向、沉陷、油类、波浪、烧伤、纵向、龟裂、其他'
    ),
    'size': fields.Float(
        required=True,
        description='病害尺寸值，必须大于0'
    ),
    'top_n': fields.Integer(
        required=False,
        default=1,
        description='返回前N个匹配模式，范围1~10，默认1'
    )
})

predict_response_model = api.model('PredictResponse', {
    'success': fields.Boolean(description='请求是否成功'),
    'message': fields.String(description='结果消息'),
    'disease_type': fields.String(description='病害类型'),
    'size': fields.Float(description='病害尺寸'),
    'unit': fields.String(description='尺寸单位'),
    'predictions': fields.List(fields.Nested(api.model('Prediction', {
        'materials': fields.List(fields.String, description='材料编号列表'),
        'material_units': fields.List(fields.String, description='材料单位列表'),
        'mean': fields.List(fields.Float, description='用量均值列表'),
        'std': fields.List(fields.Float, description='用量标准差列表'),
        'confidence': fields.Float(description='置信度(0~1)'),
        'sample_count': fields.Integer(description='历史出现次数'),
        'frequency_weight': fields.Float(description='频率权重(0~1)'),
        'area_similarity': fields.Float(description='面积相似度(0~1)'),
        'material_similarity': fields.Float(description='材料相似度(0~1)'),
        'transforms': fields.List(fields.String, description='各材料的变换类型(linear/sqrt/cbrt/square/cube)')
    })))
})

disease_types_response_model = api.model('DiseaseTypesResponse', {
    'success': fields.Boolean(description='请求是否成功'),
    'disease_types': fields.List(fields.String, description='可用病害类型列表')
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(description='服务状态'),
    'timestamp': fields.String(description='响应时间戳'),
    'version': fields.String(description='API版本')
})


def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.request_id = str(uuid.uuid4())[:8]
        g.start_time = time.time()
        g.remote_addr = request.remote_addr
        g.endpoint = request.endpoint
        
        logger.info(
            f"REQUEST_START - method={request.method} path={request.path} "
            f"query={dict(request.args)} body={request.get_json(silent=True)}",
            extra={
                'request_id': g.request_id,
                'remote_addr': g.remote_addr,
                'endpoint': g.endpoint
            }
        )
        
        try:
            response = f(*args, **kwargs)
            elapsed = round((time.time() - g.start_time) * 1000, 2)
            
            logger.info(
                f"REQUEST_END - status={response[1] if isinstance(response, tuple) else 200} "
                f"elapsed={elapsed}ms",
                extra={
                    'request_id': g.request_id,
                    'remote_addr': g.remote_addr,
                    'endpoint': g.endpoint
                }
            )
            
            return response
        except Exception as e:
            elapsed = round((time.time() - g.start_time) * 1000, 2)
            logger.error(
                f"REQUEST_ERROR - error={str(e)} elapsed={elapsed}ms",
                exc_info=True,
                extra={
                    'request_id': g.request_id,
                    'remote_addr': g.remote_addr,
                    'endpoint': g.endpoint
                }
            )
            raise
    
    return decorated_function


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not AUTH_REQUIRED:
            return f(*args, **kwargs)
        
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(
                "API_KEY_MISSING - No API key provided",
                extra={
                    'request_id': getattr(g, 'request_id', 'N/A'),
                    'remote_addr': g.remote_addr,
                    'endpoint': g.endpoint
                }
            )
            return jsonify({
                'success': False,
                'message': '缺少API密钥，请在请求头中添加 X-API-Key'
            }), 401
        
        if api_key not in VALID_API_KEYS:
            logger.warning(
                f"API_KEY_INVALID - Invalid API key: {api_key[:8]}...",
                extra={
                    'request_id': getattr(g, 'request_id', 'N/A'),
                    'remote_addr': g.remote_addr,
                    'endpoint': g.endpoint
                }
            )
            return jsonify({
                'success': False,
                'message': '无效的API密钥'
            }), 401
        
        logger.debug(
            f"API_KEY_VALIDATED - API key validated successfully",
            extra={
                'request_id': getattr(g, 'request_id', 'N/A'),
                'remote_addr': g.remote_addr,
                'endpoint': g.endpoint
            }
        )
        return f(*args, **kwargs)
    
    return decorated_function


import importlib

try:
    matching_module = importlib.import_module('5matching_system')
    Predictor = matching_module.DiseaseMaterialPredictor
    predictor = Predictor()
    logger.info(f"预测器初始化成功，加载了{len(predictor.get_disease_types())}种病害类型")
except Exception as e:
    logger.error(f"预测器初始化失败: {str(e)}", exc_info=True)
    predictor = None


@ns.route('/predict')
class Predict(Resource):
    @ns.doc('材料预测')
    @ns.expect(predict_request_model)
    @ns.marshal_with(predict_response_model)
    @log_request
    @require_api_key
    def post(self):
        """
        根据病害类型和尺寸预测材料清单
        
        根据输入的病害类型和尺寸，从案例库中智能匹配最合适的材料清单模式，
        返回预测的材料用量及置信度信息。
        """
        data = request.get_json()
        
        if not data:
            logger.warning('请求体为空')
            return {'success': False, 'message': '请求体不能为空'}, 400
        
        disease_type = data.get('disease_type')
        size = data.get('size')
        top_n = data.get('top_n', 1)
        
        if not disease_type:
            logger.warning('缺少病害类型参数')
            return {'success': False, 'message': '缺少病害类型参数'}, 400
        
        if size is None:
            logger.warning('缺少尺寸参数')
            return {'success': False, 'message': '缺少尺寸参数'}, 400
        
        try:
            size = float(size)
            if size <= 0:
                logger.warning(f'尺寸值无效: {size}')
                return {'success': False, 'message': '尺寸值必须大于0'}, 400
        except ValueError:
            logger.warning(f'尺寸值格式错误: {size}')
            return {'success': False, 'message': '尺寸值必须为数字'}, 400
        
        try:
            top_n = int(top_n)
            if top_n < 1 or top_n > 10:
                logger.warning(f'top_n值无效: {top_n}')
                return {'success': False, 'message': 'top_n必须在1~10之间'}, 400
        except ValueError:
            logger.warning(f'top_n格式错误: {top_n}')
            return {'success': False, 'message': 'top_n必须为整数'}, 400
        
        if not predictor:
            logger.error('预测器未初始化')
            return {'success': False, 'message': '服务端内部错误'}, 500
        
        result = predictor.predict(disease_type, size, top_n=top_n)
        
        if result.get('success'):
            logger.info(f'预测成功: {disease_type}, 尺寸: {size}')
        else:
            logger.warning(f'预测失败: {disease_type}, {result.get("message")}')
        
        return result


@ns.route('/disease_types')
class DiseaseTypes(Resource):
    @ns.doc('获取病害类型列表')
    @ns.marshal_with(disease_types_response_model)
    @log_request
    @require_api_key
    def get(self):
        """
        获取所有可用的病害类型
        
        返回系统支持的所有病害类型列表，用于前端下拉选择或验证。
        """
        if not predictor:
            logger.error('预测器未初始化')
            return {'success': False, 'disease_types': []}, 500
        
        disease_types = predictor.get_disease_types()
        logger.info(f'获取病害类型列表成功，共{len(disease_types)}种')
        
        return {
            'success': True,
            'disease_types': disease_types
        }


@ns.route('/health')
class HealthCheck(Resource):
    @ns.doc('健康检查')
    @ns.marshal_with(health_response_model)
    def get(self):
        """
        健康检查接口
        
        用于服务监控和负载均衡器健康探测。
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': API_VERSION
        }


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
