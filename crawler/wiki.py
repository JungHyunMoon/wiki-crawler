import os
import os
import time
from urllib.parse import urljoin

import requests
import schedule
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from VecDBLoader import embedding_text_line

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv


def login(driver, url, user_id, user_pw):
    # 웹페이지 요청
    driver.get(url)

    # 페이지 로드 대기
    time.sleep(3)  # JavaScript가 실행될 시간을 기다림

    # ID, PW 입력
    driver.find_element(By.ID, 'input-20').send_keys(user_id)
    driver.find_element(By.ID, 'input-22').send_keys(user_pw)


    # 로그인 버튼 클릭
    login_button = driver.find_element(By.XPATH,
                                       "//button[@class='mt-2 text-none v-btn v-btn--contained theme--dark v-size--large blue darken-2']")
    login_button.click()

    # 로그인 처리 대기
    time.sleep(3)


def crawl_html_by_class(driver, class_name):
    # 'contents' 클래스를 가진 div 요소 찾기
    contents_div = driver.find_element(By.CLASS_NAME, class_name)
    contents_html = contents_div.get_attribute('outerHTML')
    return contents_html


def crawl_full_page(driver):
    contents_div = driver.find_element(By.CLASS_NAME, 'v-application--wrap')
    contents_html = contents_div.get_attribute('outerHTML')
    return contents_html


def path_pointer(driver, text):
    driver.find_element(By.XPATH, f"//div[@class='v-list-item__title'][contains(text(), '{text}')]").click()
    time.sleep(0.5)


def extract_links(driver):
    html_content = crawl_html_by_class(driver, 'v-navigation-drawer__content')

    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', class_='v-list-item'):
        href = a_tag.get('href')
        full_url = urljoin("https://wiki.direa.synology.me", href)
        if full_url not in links:
            links.append(full_url)
    return links

def remove_special_characters(input_string):
    # 정규 표현식으로 알파벳과 숫자를 제외한 모든 문자를 제거
    result = input_string.replace("?", "")
    return result

def convert_html_to_md(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # 수정 기록 추출
    # aria-label이 "수정기록"인 태그 찾기
    tag = soup.find("span", text="마지막 수정:")

    # 부모 노드의 3번째 형제 노드 Text(수정 기록) 가져오기
    parent_node = tag.parent
    siblings = [s for s in parent_node.next_siblings if isinstance(s, Tag)]
    last_modified = siblings[1].get_text(strip=True)
    print(f"wiki : {last_modified}")

    # 메인 제목과 설명 추출
    main_title_element = soup.find('div', class_='headline grey--text text--darken-3')
    main_description_element = soup.find('div', class_='caption grey--text text--darken-1')

    main_title = remove_special_characters(main_title_element.get_text(strip=True)) if main_title_element else None
    main_description = main_description_element.get_text(strip=True) if main_description_element else None

    # <br> 태그를 \n으로 대체
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text_data = []

    if main_title:
        text_data.append(main_title + '\n')
    if main_description:
        text_data.append(main_description + '\n')

    soup = soup.find('div', class_='contents')

    for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'img', 'code', 'tbody', 'table']):
        text = element.get_text(separator='\n', strip=True).replace("¶", "").strip()

        if element.name == 'h1':
            text_data.append('\n\n#' + text + '\n\n')
        elif element.name == 'h2':
            text_data.append('\n\n##' + text + '\n\n')
        elif element.name == 'h3':
            text_data.append('\n\n###' + text + '\n\n')
        elif element.name == 'img':
            text = "https://wiki.direa.synology.me/" + element['src']
            text_data.append(text)
        elif element.name == 'table':
            table_text = ["\n"]
            rows = element.find_all('tr')
            headers = [th.get_text(separator='\n', strip=True) for th in rows[0].find_all(['th', 'td'])]
            headers = [header.replace("\n", "<br/>").strip() for header in headers]  # 작성자가 테이블 헤더에 br태그를 직접 넣은 경우 제거
            if headers:
                table_text.append('| ' + ' | '.join(headers) + ' |')
                table_text.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
            for tr in rows[1:]:
                row_text = []
                for td in tr.find_all('td'):
                    # td 내부의 모든 태그를 제거하고 텍스트만 추출
                    cell_text = ''.join(td.stripped_strings)
                    row_text.append(cell_text)
                table_text.append('| ' + ' | '.join(row_text) + ' |')
            text_data.append('\n'.join(table_text) + '\n')
        elif element.name == 'code':
            text_data.append(f"\n```code\n{text}\n```\n")
        else:
            if not element.find_parent(['table', 'code']) and not element.find_parent(['ul', 'ol']):
                text = text.replace("\n", "  \n")  # markdown 줄바꿈 인식을 위한 처리
                text_data.append(text)

    return main_title, text_data, last_modified

def create_session_from_driver(driver):
    session = requests.Session()
    cookies = driver.get_cookies()
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    return session


def get_cookies_dict(driver):
    cookies = driver.get_cookies()
    cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies}
    return cookie_dict


def dfs_crawl(driver, visited, crawledPages):
    while True:
        menu_div = crawl_html_by_class(driver, "__view")
        soup = BeautifulSoup(menu_div, 'html.parser')

        # <button> 요소를 제거
        for button in soup.find_all('button', string='Copy'):
            button.decompose()

        seperator_div = soup.find('div', class_='v-subheader')

        if not seperator_div:
            print("No 'v-subheader' class found.")
            break

        siblings = seperator_div.find_next_siblings('div')

        current_dir = None
        prev_div = None
        try:
            current_dir = seperator_div.find_previous_sibling('div').getText()
            prev_div = seperator_div.find_previous_sibling('div').find_previous_sibling()
            if prev_div.getText() == "sidebar.root" or prev_div.getText() == "루트":
                print("this is sidebar root directory")
                break
        except Exception as e:
            print(e)

        unvisited_sibling = None

        ############ 파일 탐색 ############
        if current_dir not in crawledPages:
            crawledPages.append(current_dir)
            # 크롤링 수행
            links = extract_links(driver)
            for link in links:
                print(link + " 페이지 크롤링 진행중")
                driver.get(link)
                contents_html = crawl_html_by_class(driver, "v-main__wrap")
                main_title, text_data, last_modified = convert_html_to_md(contents_html)
                embedding_text_line(main_title, text_data, last_modified)

        ############ 파일 탐색 ############

        ############ 경로 탐색 ############
        for sibling in siblings:
            sibling_text = sibling.get_text().strip()
            if sibling_text not in visited:
                unvisited_sibling = sibling_text
                break

        if unvisited_sibling:
            visited.add(unvisited_sibling)
            time.sleep(0.3)
            path_pointer(driver, unvisited_sibling)
            dfs_crawl(driver, visited, crawledPages)
        else:
            path_pointer(driver, prev_div.get_text().strip())
            break
        ############ 경로 탐색 ############


def save_doc(doc_title, data):
    # 텍스트 파일 경로 설정
    folder_path = os.path.join('../', 'data')

    # 폴더가 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 텍스트 파일 경로 설정
    doc_title = doc_title.replace("/", "_")
    file_path = os.path.join(folder_path, doc_title + '.txt')

    # 데이터 텍스트 파일로 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(f"{line}")

    print(f"Data saved to {file_path}")

    time.sleep(0.2)


def open_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def do_crawl():
    print("=====================================================================")
    print("===================== DIREA WIKI CRAWLING START =====================")
    print("=====================================================================")
    url = 'https://wiki.direa.synology.me/login'
    load_dotenv()
    user_id = os.getenv("USER_ID")
    user_pw = os.getenv("USER_PW")

    # Selenium WebDriver 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 브라우저 창을 열지 않고 실행
    options.add_argument('--disable-gpu')

    # WebDriver Manager를 사용하여 ChromeDriver 설치 및 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1920, 1080)
    driver.implicitly_wait(2)

    # 1. 로그인 수행
    login(driver, url, user_id, user_pw)
    print("Logged in successfully.")

    for menu_index in [1, 2, 3, 4]:
        topMenu = driver.find_element(By.XPATH,
                                      f"(//div[@class='v-list-item v-list-item--link theme--dark'][{menu_index}])")
        topMenu.click()
        time.sleep(5)

        visited = set()
        crawledPages = []
        dfs_crawl(driver, visited, crawledPages)

    # WebDriver 종료
    driver.quit()


if __name__ == "__main__":

    # 매일 자정에 한번씩 수행
    schedule.every().day.at("00:00").do(do_crawl)
    do_crawl()

    print("Schedule started.")

    while True:
        # 예약된 작업이 있는지 확인하고 실행
        schedule.run_pending()
        time.sleep(60)  # 60초마다 체크 (1분 간격)
