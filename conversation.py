import openai
import re

from dotenv import load_dotenv
import os

# .env ファイルから環境変数を読み込む
load_dotenv()

# APIキーを環境変数から取得する
openai.api_key = os.getenv("API_KEY")

# ユーザからの入力を受け取る
user_input = input("ニュース記事を入力してください：")

# プロンプト
system_prompt = """
与えられたニュース記事をユーザに対話形式で伝えるための発話計画を作成してください。
#作成する発話計画の内容
・システムの発話
・システムの発話に対しユーザから事前に聞かれそうな質問とそれに伴う回答
#作成する際の条件
・システムの発話1つに付きユーザの質問とその回答を3つ生成すること
・システムの発話は3つ生成すること
・生成する発話計画はすべて話し言葉であること
・発話計画の構成は作成する発話計画の例の同じ構成にすること
#作成する発話計画の例
システム(発話1):大谷翔平選手がロサンゼルス・ドジャースと10年7億ドルの契約を結んだって。その中の6,800万ドルは後払いで、カルフォルニア州外で受け取ると税金が節税できるんだって。
質問1.大谷選手はいくらもらえるの？      	回答:合計で約1015億円だよ。
質問2.なんで後払いにしたの？              	回答:税金の節税のためだって。
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
    for j in range(i*3, (i+3)*3):
        if j < len(qa_pairs):
            qa_list.append({
                "question": qa_pairs[j][0],
                "answer": qa_pairs[j][1]
            })
    dialogue_plan.append({
        "system_response": system_response.strip(),
        "qa_pairs": qa_list
    })

# ユーザとの対話
def user_interaction(dialogue_plan):
    for dialogue in dialogue_plan:
        # システムの発話
        print(f"\nシステム: {dialogue['system_response']}")

        for _ in range(3):
            # ユーザの質問
            user_question = input("ユーザ: ")

            # 相槌のリスト
            acknowledgements = ["へえー。", "そうなんだ。", "ふーん。", "なるほど。"]

            # ユーザが相槌を打った場合、次のシステム発話に移る
            if user_question.strip() in acknowledgements:
                break

            # 質問がない場合、次のシステム発話に移る
            if not user_question.strip():
                break

            # 質問に対する適切な回答を探す
            answered = False
            for qa_pair in dialogue["qa_pairs"]:
                if user_question in qa_pair["question"]:
                    print(f"システム: {qa_pair['answer']}")
                    answered = True
                    break

            if not answered:
                print("システム: ごめんね、その質問には答えられないよ。")
                break

# 対話開始
user_interaction(dialogue_plan)