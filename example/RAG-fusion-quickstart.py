
import os
from openai import OpenAI
import random

### RAG的fusion技术，即RAG-Fusion，是一种先进的检索增强生成（Retrieval-Augmented Generation, RAG）技术，旨在通过结合多个查询生成和互补排序融合来优化搜索结果，以更好地桥接传统搜索模式与人类查询的多维度差异。这种方法的核心在于利用大语言模型（Large Language Models, LLMs）的能力，生成与原始用户查询相关的多个查询，然后对这些查询进行相似性检索和结果排序，以提高搜索结果的相关性和准确性。


# 设置OpenAI的API密钥
os.environ["OPENAI_API_KEY"] = "sk-"

# 使用 OpenAI 的 ChatGPT 生成查询的函数
def generate_queries_chatgpt(original_query):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "您是一个乐于助人的助手，根据单个输入查询生成多个搜索查询。"},
            {"role": "user", "content": f"生成与以下内容相关的多个搜索查询：{original_query}"},
            {"role": "user", "content": "输出（4个查询）："}
        ]
    )

    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries

# 模拟向量搜索的函数，返回随机分数
def vector_search(query, all_documents):
    available_docs = list(all_documents.keys())
    random.shuffle(available_docs)
    selected_docs = available_docs[:random.randint(2, 5)]
    scores = {doc: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

# 互倒排序融合算法
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    print("初始各个搜索结果的排名：")
    for query, doc_scores in search_results_dict.items():
        print(f"对于查询 '{query}'：{doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(f"基于查询 '{query}' 中的排名 {rank}，将 {doc} 的分数从 {previous_score} 更新为 {fused_scores[doc]}")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("最终重新排序的结果：", reranked_results)
    return reranked_results

# 模拟生成输出的函数
def generate_output(reranked_results, queries):
    return f"基于 {queries} 和重新排序的文档的最终输出：{list(reranked_results.keys())}"

# 预定义的文档集合（通常来自搜索数据库）
all_documents = {
    "doc1": "气候变化及其经济影响。",
    "doc2": "由于气候变化引起的公共卫生问题。",
    "doc3": "气候变化：社会视角。",
    "doc4": "应对气候变化的技术解决方案。",
    "doc5": "应对气候变化所需的政策变革。",
    "doc6": "气候变化及其对生物多样性的影响。",
    "doc7": "气候变化：科学和模型。",
    "doc8": "全球变暖：气候变化的一个子集。",
    "doc9": "气候变化如何影响日常天气。",
    "doc10": "气候变化活动的历史。"
}

# 主函数
if __name__ == "__main__":
    original_query = "气候变化的影响"
    generated_queries = generate_queries_chatgpt(original_query)
    
    all_results = {}
    for query in generated_queries:
        ## 查询向量库或者查询web
        search_results = vector_search(query, all_documents)
        all_results[query] = search_results
    
    reranked_results = reciprocal_rank_fusion(all_results)
    
    final_output = generate_output(reranked_results, generated_queries)
    
    print(final_output)
