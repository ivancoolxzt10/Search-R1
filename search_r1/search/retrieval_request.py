import requests  # HTTP请求库

# URL for your local FastAPI server
url = "http://127.0.0.1:8000/retrieve"  # 检索服务接口地址

# Example payload
payload = {
    "queries": ["What is the capital of France?", "Explain neural networks."] * 200,  # 查询列表
    "topk": 5,  # 每个查询返回文档数
    "return_scores": True  # 是否返回分数
}

# Send POST request
response = requests.post(url, json=payload)  # 发送POST请求

# Raise an exception if the request failed
response.raise_for_status()  # 请求失败抛出异常

# Get the JSON response
retrieved_data = response.json()  # 获取JSON响应

print("Response from server:")
print(retrieved_data)  # 打印检索结果

# 文件整体功能总结：
# retrieval_request.py 用于批量测试检索服务接口，发送查询请求并打印返回结果，便于接口联调和性能验证。
