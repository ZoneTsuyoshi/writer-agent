from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# .envからAPIキーを読む
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Faceのクライアント
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=api_key
)

# 実際に問い合わせるプロンプト
prompt = """
あなたはプロのライターです。
以下のテーマに沿って、読みやすい記事を書いてください。
・小見出しを付けてください。
・日本語で書いてください。
・1000文字程度を目安にしてください。

テーマ: AIエージェントの未来
"""

# 推論実行
response = client.text_generation(
    prompt=prompt,
    max_new_tokens=800,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)

# 結果表示
print(response)

