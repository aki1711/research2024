import openai
import re
import difflib
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# .env ファイルから環境変数を読み込む
load_dotenv()


# APIキーを環境変数から取得する
openai.api_key = os.getenv("API_KEY")

# Sentence-BERTモデルをロード
model = SentenceTransformer('all-MiniLM-L6-v2')

# ユーザからの入力を受け取る
user_input = input("ニュース記事を入力してください：")


# プロンプト
system_prompt = """
あなたはユーザーにニュース記事の要約を生成した発話計画を使用して説明するシステムです。以下のフォーマットに従って発話計画を生成してください。


# 条件
- 生成する発話計画はすべて話し言葉であること。
- システムの発話は3つ生成してください。
- 各システムの発話に対し、ユーザーからの質問とその回答を5つ生成してください。
- 発話計画内で生成する質問は必ず該当するシステムの発話から生成すること。
- ニュース記事に載っていない情報は発話計画に含めないこと。


# 発話計画のフォーマット
システム(発話1): システムの最初の発話
質問1. 質問例1 回答: その質問に対する回答
質問2. 質問例2 回答: その質問に対する回答
質問3. 質問例3 回答: その質問に対する回答
質問4. 質問例4 回答: その質問に対する回答
質問5. 質問例5 回答: その質問に対する回答


システム(発話2): システムの二つ目の発話
質問1. 質問例1 回答: その質問に対する回答
質問2. 質問例2 回答: その質問に対する回答
質問3. 質問例3 回答: その質問に対する回答
質問4. 質問例4 回答: その質問に対する回答
質問5. 質問例5 回答: その質問に対する回答


システム(発話3): システムの三つ目の発話
質問1. 質問例1 回答: その質問に対する回答
質問2. 質問例2 回答: その質問に対する回答
質問3. 質問例3 回答: その質問に対する回答
質問4. 質問例4 回答: その質問に対する回答
質問5. 質問例5 回答: その質問に対する回答


#作成する発話計画の例
システム(発話1):大谷翔平選手がロサンゼルス・ドジャースと10年7億ドルの契約を結んだって。その中の6,800万ドルは後払いで、カルフォルニア州外で受け取ると税金が節税できるんだって。
質問1.大谷選手はいくらもらえるの？          回答:合計で約1015億円だよ。
質問2.なんで後払いにしたの？                 回答:税金の節税のためだって。
質問3.カルフォルニア州の税金って高いの？     回答:うん、全米で最も高いらしいよ。
システム(発話2):大谷選手は今後10年間で毎年約2.9億円を受け取るんだって。残りの約98.6億円は10年後から支払われるらしいよ。
質問1.毎年2.9億円ってのは何の金額？              回答：年俸の一部だよ。
質問2.後払いされる金額はいくらなの？             回答:約98.6億円だって。
質問3.10年後から支払われるってことは、いつから？  回答:2034年からだね。


"""


# 発話計画の生成
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_input
        },
    ],
)


# 発話計画の解析
plan = res["choices"][0]["message"]["content"]


# 発話計画を表示
print("\n生成された発話計画:")
print(plan)


# 発話計画を辞書形式に変換
# 正規表現で発話計画を解析
system_pattern = re.compile(r'システム\(発話\d+\):(.*?)\n')
qa_pattern = re.compile(r'質問\d+.(.*?)\s+回答:(.*?)\n')


system_responses = system_pattern.findall(plan)
qa_pairs = qa_pattern.findall(plan)


# 発話計画を辞書形式に変換
dialogue_plan = []
for i, system_response in enumerate(system_responses):
    qa_list = []
    for j in range(i*5, (i+1)*5): # 5つの質問生成を期待
        if j < len(qa_pairs):
            qa_list.append({
                "question": qa_pairs[j][0],
                "answer": qa_pairs[j][1]
            })
    dialogue_plan.append({
        "system_response": system_response.strip(),
        "qa_pairs": qa_list
    })


# ユーザの質問に対する適切な回答を選ぶための関数
def get_best_matching_answer(user_question, dialogue_plan):
    # ユーザの質問の埋め込みを計算
    user_embedding = model.encode([user_question])[0]

    best_match = None
    highest_similarity = -1

    for dialogue in dialogue_plan:
        for qa_pair in dialogue["qa_pairs"]:
            question_embedding = model.encode([qa_pair["question"]])[0]
            # コサイン類似度を計算
            similarity = np.dot(user_embedding, question_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(question_embedding))
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = qa_pair

    return best_match

#ユーザの発話が相槌かどうかを判定する関数を定義
def is_acknowledgement(user_responding):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは会話の相槌を判定する役割です。"},
            {"role": "user", "content": f"以下のテキストが会話の相槌かどうかを判定してください。\n\nユーザの入力: {user_responding}\n\nこの入力は会話の相槌ですか？（はい/いいえ）"}
        ],
        max_tokens=5,
        n=1,
        stop=None,
        temperature=0
    )
    return "はい" in response.choices[0].message['content'].strip()




# ユーザとの対話関数
def user_interaction(dialogue_plan):
    for dialogue in dialogue_plan:
        # システムの発話を表示
        print(f"\nシステム: {dialogue['system_response']}")


        for _ in range(3):
            # ユーザの質問を入力として受け取る
            # user_question = input("ユーザ: ")
            user_input = input("ユーザ: ")
            user_question = user_input
            user_responding = user_input

            if is_acknowledgement(user_responding):
                break

            # ユーザ質問に基づく最適な回答を取得
            best_match = get_best_matching_answer(user_input, dialogue_plan)

            # 最適な質問とその回答を表示
            if best_match:
                print(f"システム: {best_match['answer']}")
            else:
                print("システム: すみません、適切な回答が見つかりませんでした。")

# 対話開始
user_interaction(dialogue_plan)
