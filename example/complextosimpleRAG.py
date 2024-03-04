from fastapi import FastAPI, HTTPException
from typing import List, Dict
import asyncio
import logging
import httpx  # 用于异步HTTP请求

app = FastAPI()
# OpenAI API密钥的占位符
OPENAI_API_KEY = "your_openai_api_key_here"

async def decompose_complex_question(question: str) -> List[str]:
    """
    将复杂问题分解为简单问题列表。
    Args:
        question (str): 待分解的复杂问题。
    Returns:
        List[str]: 分解后的简单问题列表。
    """
    # 系统消息，用于指导问题的分解
    system_message = {"role": "system", "content": "选择最好的问题来分解复杂问题。目标是全面性。"}
    # 用户消息，即待分解的复杂问题
    user_message = {"role": "user", "content": question}
    messages = [system_message, user_message]

    # 示例中的模拟响应，实际应用中需要替换为API调用的结果
    response = {"choices": [{"message": {"content": "问题1\n问题2\n问题3"}}]}
    decomposed_questions = response['choices'][0]['message']['content'].strip().split('\n')

    return decomposed_questions

async def specific_query(query_text: str) -> Dict[str, str]:
    """
    向GPT模型查询特定问题的答案。
    
    Args:
        query_text (str): 需要查询的问题。
    
    Returns:
        Dict[str, str]: 包含问题及模型响应的字典。
    """
    try:
        # 使用问题作为GPT模型的提示
        response_text = await get_gpt_response(query_text)
        return {"question": query_text, "answer": response_text}
    except HTTPException as e:
        # 若发生HTTP异常，返回错误消息
        logging.error(f"查询GPT模型时出错: {e.detail}")
        return {"question": query_text, "answer": "从GPT模型获取响应失败。"}

@app.get("/query/{query_text}")
async def query_category(query_text: str):
    """
    处理查询请求，将复杂问题分解后查询每个简单问题的答案，并汇总。
    
    Args:
        query_text (str): 查询文本。
    
    Returns:
        Dict[str, str]: 包含最终答案的字典。
    """
    # 将复杂问题分解为简单问题
    simple_questions = await decompose_complex_question(query_text)
    # 对每个简单问题进行查询
    simple_answers_tasks = [specific_query(sq) for sq in simple_questions]
    simple_answers_results = await asyncio.gather(*simple_answers_tasks)
    simple_answers = [{"question": sq, "answer": result['answer']} for sq, result in zip(simple_questions, simple_answers_results)]

    # 汇总答案
    final_answer = await aggregate_answers(query_text, simple_answers)

    logging.info(f"简单问题: {simple_questions}")
    logging.info(f"简单答案: {simple_answers}")
    logging.info(f"最终答案: {final_answer}")

    return {"response": final_answer}

async def aggregate_answers(complex_question: str, simple_qas: List[Dict[str, str]]):
    """
    根据简单问题及答案汇总出复杂问题的答案。
    
    Args:
        complex_question (str): 复杂问题。
        simple_qas (List[Dict[str, str]]): 简单问题及答案的列表。
    
    Returns:
        str: 汇总后的答案。
    """
    # 构造提示文本，包括复杂问题及所有简单问题和答案
    qa_text = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in simple_qas])
    prompt = f"根据以下问题和答案，综合出对以下复杂问题的解释: {complex_question}"

    # 示例中的模拟聚合逻辑，实际应用中需要替换为实际的聚合逻辑
    aggregated_answer = "基于提供的问题与答案模拟的综合解释"

    return aggregated_answer

async def get_gpt_response(prompt: str) -> str:
    """
    给定提示文本，从GPT模型获取响应。
    
    Args:
        prompt (str): 提示文本。
    
    Returns:
        str: 模型的响应文本。
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": 150,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.openai.com/v1/completions", json=data, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="GPT模型请求失败")
    return response.json()["choices"][0]["text"].strip()

# 示例端点的使用
@app.get("/gpt_query/{query_text}")
async def gpt_query_endpoint(query_text: str):
    """
    GPT查询端点，返回GPT模型对特定查询的响应。
    
    Args:
        query_text (str): 查询文本。
    
    Returns:
        Dict[str, str]: 包含响应或错误信息的字典。
    """
    try:
        response_text = await get_gpt_response(query_text)
        return {"response": response_text}
    except HTTPException as e:
        return {"error": str(e.detail)}
