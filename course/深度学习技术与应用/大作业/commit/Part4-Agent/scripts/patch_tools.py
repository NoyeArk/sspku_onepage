"""
DeepDiver V2 工具实现补丁脚本
为 mcp_tools.py 中的 _generic_search 和 _content_extractor 提供实际实现。
使用 Serper API 实现搜索，使用 Jina Reader 实现网页内容提取。
"""

import os
import re
import shutil

SEARCH_IMPL = '''    def _generic_search(self, query: str, max_results: int, config: Dict[str, Any]) -> MCPToolResult:
        """使用 Serper API 实现网页搜索"""
        try:
            url = config.get('base_url', 'https://google.serper.dev/search')
            api_keys = config.get('api_keys', [])
            if not api_keys:
                return MCPToolResult(success=False, error="No search API keys configured")

            headers = {
                'X-API-KEY': random.choice(api_keys),
                'Content-Type': 'application/json'
            }
            payload = json.dumps({"q": query, "num": max_results})
            response = requests.post(url, data=payload, headers=headers, timeout=30)
            response.raise_for_status()
            results = response.json()

            search_results = {
                "organic": [
                    {
                        "title": r.get("title", ""),
                        "link": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                        "date": r.get("date", "unknown")
                    }
                    for r in results.get("organic", [])
                ]
            }
            return MCPToolResult(success=True, data=search_results)
        except Exception as e:
            return MCPToolResult(success=False, error=f"Generic search failed: {e}")
'''

CRAWLER_IMPL = '''    def _content_extractor(self, url: str, max_tokens: int, config: Dict[str, Any]) -> MCPToolResult:
        """使用 Jina Reader 实现网页内容提取"""
        max_retry_num = 3
        sleep_time = 3
        retry_num = 0
        while True:
            retry_num += 1
            try:
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    "Accept": "text/plain",
                    "X-Return-Format": "text",
                }
                api_keys = config.get('api_keys', [])
                if api_keys and api_keys[0] != 'default_key':
                    headers["Authorization"] = f"Bearer {random.choice(api_keys)}"

                resp = requests.get(jina_url, headers=headers, timeout=60)
                resp.raise_for_status()
                content = resp.text

                if max_tokens and len(content.split()) > max_tokens:
                    words = content.split()[:max_tokens]
                    content = ' '.join(words) + '...'

                return MCPToolResult(success=True, data=content)
            except Exception as e:
                if retry_num >= max_retry_num:
                    return MCPToolResult(success=False, error=f"Content extractor failed after {max_retry_num} retries: {e}")
                else:
                    import time as _time
                    _time.sleep(sleep_time)
'''


def patch_mcp_tools(deepdiver_v2_path: str):
    """
    替换 mcp_tools.py 中的 _generic_search 和 _content_extractor 实现。
    会先备份原文件为 mcp_tools.py.bak。
    """
    mcp_tools_path = os.path.join(deepdiver_v2_path, "src", "tools", "mcp_tools.py")

    if not os.path.exists(mcp_tools_path):
        raise FileNotFoundError(f"找不到文件: {mcp_tools_path}")

    backup_path = mcp_tools_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(mcp_tools_path, backup_path)
        print(f"已备份原始文件到: {backup_path}")
    else:
        # 备份已存在，说明之前打过补丁；从备份恢复原始文件以确保干净的起点
        shutil.copy2(backup_path, mcp_tools_path)
        print(f"已从备份恢复原始文件: {backup_path}")

    with open(mcp_tools_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换 _generic_search 方法（lookahead 需同时匹配 @decorator 和 def，避免吞掉下一个方法的装饰器）
    search_pattern = r'(    def _generic_search\(self.*?\n)(.*?)(?=\n    (?:@|def ))'
    match = re.search(search_pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + SEARCH_IMPL + '\n' + content[match.end():]
        print("✅ 已替换 _generic_search 实现（Serper API）")
    else:
        print("⚠️ 未找到 _generic_search 方法，跳过")

    # 替换 _content_extractor 方法
    crawler_pattern = r'(    def _content_extractor\(self.*?\n)(.*?)(?=\n    (?:@|def ))'
    match = re.search(crawler_pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + CRAWLER_IMPL + '\n' + content[match.end():]
        print("✅ 已替换 _content_extractor 实现（Jina Reader）")
    else:
        print("⚠️ 未找到 _content_extractor 方法，跳过")

    with open(mcp_tools_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # 验证关键结构完整性
    if '@staticmethod' in content and '_extract_publication_date_from_html' in content:
        print("✅ @staticmethod 装饰器完整保留")
    else:
        print("❌ 警告: @staticmethod 装饰器可能丢失，请检查!")

    print(f"✅ 补丁完成: {mcp_tools_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch DeepDiver V2 mcp_tools.py")
    parser.add_argument("--deepdiver-path", required=True, help="deepdiver_v2 目录路径")
    args = parser.parse_args()
    patch_mcp_tools(args.deepdiver_path)
