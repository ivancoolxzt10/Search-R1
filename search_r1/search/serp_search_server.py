import os  # 操作系统相关库
import requests  # HTTP请求库
from fastapi import FastAPI  # Web框架
from pydantic import BaseModel  # 数据模型校验
from typing import List, Optional, Dict  # 类型注解
from concurrent.futures import ThreadPoolExecutor  # 线程池并发库
import argparse  # 命令行参数解析库
import uvicorn  # ASGI服务器

parser = argparse.ArgumentParser(description="Launch online search server.")
parser.add_argument('--search_url', type=str, required=True, 
                    help="URL for search engine (e.g. https://serpapi.com/search)")  # SerpAPI搜索接口
parser.add_argument('--topk', type=int, default=3,
                    help="Number of results to return per query")  # 每次查询返回结果数
parser.add_argument('--serp_api_key', type=str, default=None,
                    help="SerpAPI key for online search")  # SerpAPI密钥
parser.add_argument('--serp_engine', type=str, default="google",
                    help="SerpAPI engine for online search")  # 搜索引擎类型
args = parser.parse_args()

# --- Config ---
class OnlineSearchConfig:
    """
    在线搜索配置类。
    search_url: SerpAPI接口地址
    topk: 返回结果数量
    serp_api_key: SerpAPI密钥
    serp_engine: 搜索引擎类型
    """
    def __init__(
        self,
        search_url: str = "https://serpapi.com/search",
        topk: int = 3,
        serp_api_key: Optional[str] = None,
        serp_engine: Optional[str] = None,
    ):
        self.search_url = search_url
        self.topk = topk
        self.serp_api_key = serp_api_key
        self.serp_engine = serp_engine


# --- Online Search Wrapper ---
class OnlineSearchEngine:
    """
    在线搜索引擎主类，负责SerpAPI搜索、结果处理。
    """
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str):
        """
        单条查询接口，调用SerpAPI。
        """
        params = {
            "engine": self.config.serp_engine,
            "q": query,
            "api_key": self.config.serp_api_key,
        }
        response = requests.get(self.config.search_url, params=params)
        return response.json()

    def batch_search(self, queries: List[str]):
        """
        批量查询接口，支持多线程并发。
        """
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._search_query, queries):
                results.append(self._process_result(result))
        return results

    def _process_result(self, search_result: Dict):
        """
        处理SerpAPI返回结果，提取标题和摘要。
        """
        results = []
        
        answer_box = search_result.get('answer_box', {})
        if answer_box:
            title = answer_box.get('title', 'No title.')
            snippet = answer_box.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            })

        organic_results = search_result.get('organic_results', [])
        for _, result in enumerate(organic_results[:self.config.topk]):
            title = result.get('title', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            })

        related_results = search_result.get('related_questions', [])
        for _, result in enumerate(related_results[:self.config.topk]):
            title = result.get('question', 'No title.')  # question is the title here
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            })

        return results


# --- FastAPI Setup ---
app = FastAPI(title="Online Search Proxy Server")

class SearchRequest(BaseModel):
    queries: List[str]  # 查询列表

# Instantiate global config + engine
config = OnlineSearchConfig(
    search_url=args.search_url,
    topk=args.topk,
    serp_api_key=args.serp_api_key,
    serp_engine=args.serp_engine,
)
engine = OnlineSearchEngine(config)

# --- Routes ---
@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    """
    FastAPI接口：批量在线检索。
    """
    results = engine.batch_search(request.queries)
    return {"result": results}

## return {"result": List[List[{'document': {"id": xx, "content": "title" + \n + "content"}, 'score': xx}]]}

if __name__ == "__main__":
    # 启动服务，默认监听 http://0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 文件整体功能总结：
# serp_search_server.py 实现了基于SerpAPI的在线检索服务，支持批量查询、结果格式化，便于RAG、问答等场景的高效集成。
