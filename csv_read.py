import pandas as pd
from sqlalchemy import create_engine

db_uri = "postgresql+psycopg2://user_kotaro:fractal_first_test@localhost:5432/first_test"
engine = create_engine(db_uri)
conn = engine.connect()
search_sql = f"""
SELECT embedding FROM "80kdatatest" WHERE item_id = 'ITEM000003'
"""
result_df = pd.read_sql(search_sql, conn)
print(result_df)
# # CSVを読み込み
# csv_path = "80kitem_list.csv"
# df = pd.read_csv(csv_path)

# # テーブルを作成して挿入（既存テーブルがあれば上書き）
# df.to_sql("80kdatatest", engine, if_exists='replace', index=False)

# print(f"✅ CSVデータを挿入しました")