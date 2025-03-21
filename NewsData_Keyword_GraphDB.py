import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from konlpy.tag import Komoran
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HEADERS = {'User-Agent': 'Mozilla/5.0'}
URI ="bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "chlwhdgus12?"

def fetch_news(date):
    url = f"https://news.naver.com/main/ranking/popularDay.nhn?date={date}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    rankingnews = soup.find_all(class_='rankingnews_box')

    news_list = []
    for item in rankingnews:
        media = item.a.strong.text.strip()
        news = item.find_all(class_="list_content")
        for new in news:
            news_list.append({
                'media': media,
                'src': f"https://news.naver.com/{new.a['href']}",
                'title': new.a.text.strip(),
                'date': date
            })
    return news_list

def extract_keywords(df):
    komoran = Komoran()
    df['keyword'] = df['title'].apply(lambda title: ', '.join([noun for noun in komoran.nouns(title) if len(noun) > 1]))
    return df

def clean_title(title):
    return re.sub(r'[^a-zA-Z0-9ㄱ-ㅣ가-힣]','', title)

def save_to_neo4j(df):
    def add_article_query(tx, title, date, media, keyword):
        tx.run("""
            merge (a:News {title: $title, date: $date, media: $media, keyword:$keyword})
            """, title=title, date=date, media=media,keyword=keyword)
    def add_media_query(tx):
        tx.run("""
            match (a:News)
            merge (b:NewsMedia {name: a.media})
            merge (a)<-[r:Print]-(b)
            """)

    def add_keyword_query(tx):
        tx.run("""
            match (a:News)
            unwind split(a.keyword, ', ') as k
            merge (b:NewsKeyword {name: k})
            merge (a)-[r:Consists_of]->(b)
            """)

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        for _, row in df.iterrows():
            session.execute_write(add_article_query, title=row['clean_title'],date=row['date'], media=row['media'], keyword=row['keyword'])
        session.execute_write(add_media_query)
        session.execute_write(add_keyword_query)
    driver.close()

def main():
    logging.info("뉴스 데이터를 수집합니다.")
    start_date, end_date = 20241004, 20241020
    all_news = []

    for date in range(start_date, end_date +1):
        try :
            news = fetch_news(str(date))
            all_news.extend(news)
            logging.info(f"{date}의 뉴스를 성공적으로 수집했습니다.")
        except Exception as e:
            logging.error(f"{date}의 뉴스를 수집 중 오류 발생: {e}")

    df = pd.DataFrame(all_news)

    logging.info("키워드 추출 중입니다.")
    df = extract_keywords(df)

    logging.info("뉴스 제목 정리 중입니다.")
    df['clean_title'] = df['title'].apply(clean_title)

    logging.info("Neo4j에 데이터를 저장합니다.")
    save_to_neo4j(df)

    logging.info("작업이 완료되었습니다.")

if __name__=="__main__":
    main()
