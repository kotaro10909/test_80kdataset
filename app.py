import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from streamlit.logger import get_logger
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, List
from langchain_core.retrievers import BaseRetriever  # LangChain最新版用
from pydantic import BaseModel
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from typing import List, Any
from pgvector.psycopg2 import register_vector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
logger = get_logger(__name__)
class Embedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)


class PostgresVectorRetriever(BaseRetriever, BaseModel):
    conn: Any
    table_name: str
    embedding_function: Any  # GoogleEmbedding など
    k: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)  # ← Pydanticの設定

    def get_embedding(self, input_texts: list[str]) -> list[list[float]]:
        return self.embedding_function.embed_documents(input_texts)
    
    def update_vector(self) -> List[Document]:
        # 1. 未登録データにembeddingを付与して更新
        df = pd.read_sql(
            f'SELECT item_id, item_name, item_detail FROM "{self.table_name}" WHERE embedding IS NULL LIMIT 10',
            self.conn
        )

        if not df.empty:
            for i, row in df.iterrows():
                try:
                    # 1件ずつ埋め込み（Embeddingクラスを使う）
                    text = f"{row['item_name']} {row['item_detail']}"
                    vec = self.embedding_function.embed_query(text)

                    print(f"{i}: {row['item_name']} -> {vec[:5]}")  # デバッグ出力（先頭5次元のみ）

                    update_query = text(f"""
                        UPDATE "{self.table_name}" SET embedding = :embedding WHERE item_id = :item_id
                    """)
                    self.conn.execute(update_query, {
                        "embedding": vec,
                        "item_id": row["item_id"]
                    })
                    self.conn.commit()

                except Exception as e:
                    print(f"Embedding error at index {i}: {e}")
                    continue




    def get_relevant_documents(self, query: str) -> List[Document]:
        
        #  検索クエリをベクトル化
        query_vector = self.embedding_function.embed_query(query)

        ## クエリベクトルをPostgreSQL形式に変換
        # query_vector: List[float]
        vector_str = f"[{', '.join(map(str, query_vector))}]"

        search_sql = f"""
            SELECT item_name, item_detail,
                1 - (embedding <-> '{vector_str}'::vector) AS similarity
            FROM "{self.table_name}"
            ORDER BY embedding <-> '{vector_str}'::vector
            LIMIT {self.k}
        """
        result_df = pd.read_sql(search_sql, self.conn)

        # 結果を Document にして返す
        return [
            Document(
                page_content=row["item_detail"],
                metadata={
                    "title": row["item_name"],
                    "similarity": row["similarity"]
                }
            )
            for _, row in result_df.iterrows()
        ]




embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Prompt テンプレート
prompt_template = """
質問の内容と一致するキーワードを背景情報の中から回答を関連度の高い順番で3つ候補出してください（先頭の文字は、1. 2. 3.と続くように）。
同じ行の内容も表示する（例　　1.商品1：コンクリート）。
# 背景情報
{context}

# 質問
{question}
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.0,
    max_retries=2,
)


# ページ設定
st.set_page_config(
    page_title="Vector Database Demo",
    page_icon="🔍",
    layout="wide"
)
def run_qa_from_db(search_text):
    raw_conn = psycopg2.connect("dbname=first_test user=user_kotaro password=fractal_first_test")
    register_vector(raw_conn)
    raw_conn.close()
    db_uri = "postgresql+psycopg2://user_kotaro:fractal_first_test@localhost:5432/first_test"
    engine = create_engine(db_uri)
    conn = engine.connect()
    table_name = "80kdatatest"
    query = f'SELECT item_id,item_name,item_detail FROM "{table_name}" LIMIT 1000'
    df = pd.read_sql(query, conn)
    if df.empty:
        st.info("データが登録されていません")
        return
    # CSVとして保存（ファイルパスは適宜変更）
    # csv_path = "output_documents.csv"
    # df.to_csv(csv_path, index=False, encoding="utf-8")
    # DataFrameの内容をCSV風の文字列に変換（ヘッダー除外）
    csv_text = "\n".join([
        f"{row['item_id']} / {row['item_name']} / {row['item_detail']}"
        for _, row in df.iterrows()
    ])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=1,
        length_function=len,
        separators=["\n"],  # 柔軟に対応
    )

    all_chunks = []

    # csv_files = glob.glob('output_documents.csv')
    # for csv_path in csv_files:
    #     with open(csv_path, newline="", encoding="utf-8-sig") as csvfile:
    #         reader = csv.reader(csvfile)
    #         headers = next(reader)
    #         csv_text = "\n".join([" / ".join(row) for row in reader])

    csv_chunks = text_splitter.split_text(csv_text)
    for i, chunk in enumerate(csv_chunks):
        doc = Document(page_content=chunk, metadata={"source_csv": "postgresDB", "chunk_index": i})
        all_chunks.append(doc)

    print(f"✅ CSVチャンク数: {len(csv_chunks)}")
    # FAISSベースのインメモリベクトルストア作成
    # vectorstore = FAISS.from_documents(all_chunks, embedding)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    retriever = PostgresVectorRetriever(
    conn=conn,
    embedding_function=embedding,  # LangChainのEmbedding（例: GoogleEmbeddingなど）
    table_name=table_name,
    k=10
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    retriever.update_vector()
    docs = retriever.get_relevant_documents("ベクトル詳細が知りたい")
    for doc in docs:
        print(doc.metadata, doc.page_content[:50])
    # response = qa.invoke(search_text)
    # 回答の表示
    with st.chat_message("assistant"):
        st.markdown("お探しのキーワードはこちらですか？")
    with st.chat_message("assistant"):
        for idx, doc in enumerate(docs):
            with st.expander(f"候補 {idx+1}（類似度: {doc.metadata['similarity']:.3f}）"):
                st.markdown(f"**タイトル**: {doc.metadata['title']}")
                st.markdown(doc.page_content[:100] + "...")

def get_texttosql(question):
    # PostgreSQLへの接続
    try:
        pg = st.secrets["connections"]["postgresql"]
        db_uri = f"{pg['dialect']}+psycopg2://{pg['username']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
        db = SQLDatabase.from_uri(db_uri, include_tables=["80kdatatest"])

        # LLMの設定（Gemini）
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
        )

        # SQL Agentの作成
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True)

        # 質問の実行
        request_sql = f"""
            {question}というキーワードに関連する(誤字脱字を考慮する)商品を最大3件まで選び、item_id, item_name, item_detailを含む結果を出力してください。
            SQLは必ず SELECT 文のみを使用すること。該当するデータがないときは、「該当データがありません」と回答してください。
            出力は以下の形式にしてください：

            1. <item_id> / <item_name> / <item_detail>  
            2. <item_id> / <item_name> / <item_detail>  
            3. <item_id> / <item_name> / <item_detail>
            """
        response = agent_executor.run(request_sql)
        return response
    except Exception as e:
                    print(f"error occured {e}")
                    return None


def get_embedding(text):
    return embedding.embed_query(text)

# データベース接続関数
def get_connection():
    return st.connection('postgresql', type='sql')

# メインアプリケーション
def main():
    st.title("📊 Postgres AI Search")
    
    # サイドバーでの操作選択
    operation = st.sidebar.selectbox(
        "操作を選択",
        ["データ表示", "データ追加", "ベクトル検索"]
    )
    
    if operation == "データ表示":
        show_data()
    elif operation == "データ追加":
        add_data()
    elif operation == "ベクトル検索":
        vector_search()

def show_data():
    st.header("📋 登録データ一覧")
    
    conn = get_connection()

    # データ取得
    table_name = "80kdatatest"
    query = f'SELECT item_id,item_name,item_detail FROM "{table_name}" LIMIT 100'
    df = conn.query(query, ttl=0)
    
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("データが登録されていません")

def add_data():
    st.header("➕ データ追加")
    
    # 入力フォーム
    with st.form("data_form"):
        title = st.text_input("タイトルを入力")
        content = st.text_area("コンテンツ情報を入力")
        submitted = st.form_submit_button("登録")
        
        if submitted and content:
            conn = get_connection()
            
            # サンプルとして、ランダムな1536次元ベクトルを生成
            # 実際のアプリケーションでは、適切なエンベッディングモデルを使用する
            embedding = get_embedding(title + " " + content)
            
            # データ登録
            query = text("""
            INSERT INTO documents (title, content, embedding)
            VALUES (:title, :content, :embedding)
            """)
            params = {"title": title, "content": content, "embedding": embedding}
            
            try:
                with conn.session as session:
                    session.execute(query, params)
                    session.commit()
                st.success("データを登録しました")
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

def vector_search():
    st.header("🔍 ベクトル検索")
    search_text = st.text_input("質問を入力してください")
    if st.button("検索") and search_text:
        result = get_texttosql(search_text)
        # with st.chat_message("assistant"):
        #     st.markdown("お探しのキーワードはこちらですか？")
        with st.chat_message("assistant"):
            if result:
                st.markdown("お探しのキーワードはこちらですか？")
                st.markdown(result)
            else:
                st.markdown("お探しのキーワードは見つかりませんでした")

       

# アプリケーション実行
if __name__ == "__main__":
    main()


# from langchain.chains import RetrievalQA
# from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# import streamlit as st
# from langchain.prompts import ChatPromptTemplate
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# from langchain.prompts import PromptTemplate
# import psycopg2
# import numpy as np
# from langchain.vectorstores.pgvector import PGVector

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBnmd9ILsLXBucQM9K9VxUYcVS9ZfI_v60"

# def app_login():
#     # 未ログイン時の表示
#     if not st.user.is_logged_in:
#         if st.button("Googleでログイン"):
#              st.cache_data.clear()
#              st.login()
#              st.rerun()

# def user_check():
#     allowed_emails = st.secrets["access"]["allowed_emails"]
#     if st.user.email not in allowed_emails:
#         st.error("このユーザーにはアクセス権限がありません。")
#         if st.button("別のアカウントでログイン"):
#              st.logout()
#              st.cache_data.clear()
#              st.rerun()
#         return True
#     return False

# def app_logout():
#         if st.button("ログアウト"):
#             st.logout()
#             st.cache_data.clear()
#             st.rerun()

# # def load_db(embeddings):
# #     return FAISS.load_local('faiss_store', embeddings, allow_dangerous_deserialization=True)

# def load_db(embeddings):
#     CONNECTION_STRING = "postgresql+psycopg2://user_kotaro:fractal_first_test@localhost:5432/first_test"
#     COLLECTION_NAME = "Shohin"

#     return PGVector(
#         collection_name=COLLECTION_NAME,
#         connection_string=CONNECTION_STRING,
#         embedding_function=embeddings,
#     )


# def init_page():
#     st.set_page_config(
#         page_title='FRACTAL AI SEARCH',
#         page_icon='🧑‍💻',
#     )


# def main():
#     init_page()
#   # if not st.user.is_logged_in:
#   #   app_login()
#   # else:
#   #   if user_check():
#   #       return
#   #   app_logout()
#     # DB接続情報
#     conn = psycopg2.connect(
#         host="localhost",
#         dbname="first_test",
#         user="user_kotaro",
#         password="fractal_first_test",
#         port=5432,
#     )

#     cur = conn.cursor()

#     # 埋め込み生成モデル
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # 商品データ取得
#     cur.execute("SELECT id, title, description FROM Shohin")
#     rows = cur.fetchall()
#     print(rows)
#     for row in rows:
#         id_, title, description = row
#         text = title + " " + description
#         vector = embeddings.embed_query(text)  # list[float]
#         # 直接 vector を渡す（文字列にしない）
#         cur.execute(
#             "UPDATE Shohin SET embedding = %s WHERE id = %s",
#             (vector, id_)
#         )


#     conn.commit()
#     cur.close()
#     conn.close()
#     db = load_db(embeddings)

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash-lite",
#         temperature=0.0,
#         max_retries=2,
#     )

#     # オリジナルのSystem Instructionを定義する

#     # prompt_template = """
#     # あなたは、「北辰物産」という団体専用のチャットボットです。
#     # 背景情報を参考に、質問に対して団体の人間になりきって、質問に回答してくだい。解答の最後で、参照したURLが分かればそれを答えてください。
#     # 以下の背景情報を参照してください。背景情報と関係のある質問には答えてください。
#     # 情報がなければ、その内容については言及しないでください。
#     # # 背景情報
#     # {context}

#     # # 質問
#     # {question}"""
#     prompt_template = """
#     あなたは、「フラクタル建築」という団体専用のチャットボットです。
#     背景情報を参考に、質問に対して団体の人間になりきって、質問に回答してくだい。回答情報と対応するDBのカラムの情報も加えてください。
#     以下の背景情報を参照してください。背景情報と関係のある質問には答えてください。
#     情報がなければ、その内容については言及しないでください。
#     # 背景情報
#     {context}

#     # 質問
#     {question}"""
#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}# システムプロンプトを追加
#     )

#     # 上半分に説明文とリンクを配置
#     # HTMLタグをそのまま使ってスタイルを適用
#     st.markdown("""
#         <style>
#         .chat-box1 {
#             overflow-y: auto;
#             border: 2px solid #ddd;
#             padding: 20px;
#             border-radius: 8px;
#             background-color: none;
#             margin-bottom: 20px;
#             text-align: center;
#             background-color = #f5f5f5;
#             font: monospace;
#             color: #333333;
#         }
#         .chat-box1 h1{
#             color: #00536D;
#         }
#         </style>
#         """, unsafe_allow_html=True)

#     # 直接HTML要素を表示
#     st.markdown("""
#         <div class="chat-box1">
#             <h1>FRACTAL AI SEARCH</h1>
#             <p>このAIチャットアプリは、社内情報を検索するためのものです。</p>
#             <p>フラクタル建築に関する質問以外にはお答えできません。（実際の企業名や社内情報は入っていません。）</p>
#             <a href="#">社内ポータルサイト</a>
#         </div>
#     """, unsafe_allow_html=True)

#     # 下半分にチャット画面を配置
#     if "messages" not in st.session_state:
#       st.session_state.messages = []
#     if user_input := st.chat_input('質問しよう！'):
#         # 以前のチャットログを表示
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#         print(user_input)
#         with st.chat_message('user'):
#             st.markdown(user_input)
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message('assistant'):
#             with st.spinner('Gemini is typing ...'):
#                 response = qa.invoke(user_input)
#             st.markdown(response['result'])
#             #参考元を表示
#             doc_urls = []
#             doc_pdfs = []
#             for doc in response["source_documents"]:
#                 #既に出力したのは、出力しない
#                 if "source_url" in doc.metadata and doc.metadata["source_url"] not in doc_urls:
#                     doc_urls.append(doc.metadata["source_url"])
#                     st.markdown(f"参考元：{doc.metadata['source_url']}")
#                 elif "source_pdf" in doc.metadata and doc.metadata["source_pdf"] not in doc_pdfs:
#                     doc_pdfs.append(doc.metadata["source_pdf"])
#                     st.markdown(f"参考元：{doc.metadata['source_pdf']}")
#         st.session_state.messages.append({"role": "assistant", "content": response["result"]})


# if __name__ == "__main__":
#   main()