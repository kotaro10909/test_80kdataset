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
    raw_conn = psycopg2.connect("x")
    register_vector(raw_conn)
    raw_conn.close()
    db_uri = "postgresql+psycopg2://xx"
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


