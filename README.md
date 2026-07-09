# 材料清单模式识别

道路病害材料用量预测系统，基于历史案例库的智能匹配算法，根据病害类型和尺寸预测所需的材料清单及用量。系统通过线性性验证自动为每种材料选择最优的数学变换（线性、二次根、三次根、二次方、三次方），提供 RESTful API 接口，支持 Swagger 文档、API Key 认证和增强日志，可通过 Docker 快速部署。

## 核心特性

- **多变换拟合**：通过 `4verify_linear.py` 验证每种材料的用量与尺寸的最优函数关系，不再强制假设线性
- **外挂配置**：病害单位、材料单位、变换类型统一写入 `config.yaml`，无需改代码即可调整
- **案例匹配算法**：基于历史施工案例库，通过频率加权、面积相似度、材料组合相似度等多维度评分
- **RESTful API**：提供标准化的 HTTP 接口，支持 JSON 格式交互
- **Swagger 文档**：内置交互式 API 文档，支持在线调试
- **API Key 认证**：保护接口安全，防止未授权调用
- **增强日志**：每个请求携带唯一 request_id，记录耗时、客户端 IP 等信息
- **Docker 部署**：一键容器化部署，适合生产环境

## 项目结构

```
yanghu/
├── app.py                    # API 服务主程序（Flask + flask-restx）
├── 5matching_system.py       # 核心匹配算法（DiseaseMaterialPredictor）
├── 4verify_linear.py         # 线性性验证脚本（生成 config.yaml 和验证报告）
├── config.yaml               # 外挂配置文件（病害单位、材料单位、变换类型）
├── dataset/
│   └── lumian/               # 原始案例数据（按病害类型分文件，共13种）
│       ├── 车辙.jsonl
│       ├── 坑槽.jsonl
│       └── ...
├── .env                      # 环境变量配置文件
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 镜像构建文件
├── docker-compose.yml        # Docker Compose 编排文件
├── app.log                   # 运行日志（自动生成）
└── doc/                      # 项目文档
    ├── api_documentation.md
    ├── api_interface_spec.md
    └── linearity_report.md   # 线性性验证报告
```

## 许可证

本项目仅供学习和研究使用。
