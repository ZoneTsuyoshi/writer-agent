import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()

# Hugging FaceのAPIキー
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Llama 3 7B Instructを呼び出す設定
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=api_key,
    task="text-generation",
)

# プロンプト（記事生成指示）
prompt = PromptTemplate.from_template(
    """
あなたはプロのライターです。
以下のテーマに沿って、読みやすい記事を書いてください。
・小見出しを付けてください。
・日本語で書いてください。
・1000文字程度を目安にしてください。

テーマ: {theme}
"""
)

# チェーンを作る
chain = LLMChain(llm=llm, prompt=prompt)

# ユーザーからテーマを受け取って生成
theme = "AIエージェントの未来"
output = chain.invoke({"theme": theme})

print(output["text"])

