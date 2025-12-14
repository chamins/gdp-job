import streamlit as st
import pandas as pd
import re
from pathlib import Path
from io import BytesIO
import requests
from urllib.parse import quote_plus
import smtplib
from email.message import EmailMessage

# Optional heavy imports
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    import docx
except Exception:
    docx = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
except Exception:
    TfidfVectorizer = None
    linear_kernel = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    S2_AVAILABLE = True
    S2_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    S2_AVAILABLE = False
    S2_MODEL = None


st.set_page_config(page_title='잡데렐라', page_icon='🫧', layout='wide')


def local_css():
    css = '''
    <style>
    .stApp { background: linear-gradient(180deg, #FFF8FB 0%, #FFFDF8 100%);} 
    .big-logo {font-size:42px; font-weight:700; color:#FF6FA3;}
    .subtitle {color:#9B6DFF; font-size:18px}
    .card {background:#ffffff80; border-radius:14px; padding:12px;}
    .rounded {border-radius:12px}
    .pastel-btn {background: linear-gradient(90deg,#FFD1DC,#D6C3FF); color:#222; border:none; padding:8px 12px; border-radius:10px}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data
def load_jobs():
    fn = Path(__file__).parent / 'data' / 'jobs_mock.csv'
    if fn.exists():
        return pd.read_csv(fn)
    else:
        return pd.DataFrame(columns=['company','title','region','salary','exp_level','description','url'])


def extract_text_from_pdf(file_bytes):
    if PdfReader is None:
        return ''
    try:
        reader = PdfReader(BytesIO(file_bytes))
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or '')
        return '\n'.join(text)
    except Exception:
        return ''


def extract_text_from_docx(file_bytes):
    if docx is None:
        return ''
    try:
        doc = docx.Document(BytesIO(file_bytes))
        return '\n'.join([p.text for p in doc.paragraphs])
    except Exception:
        return ''


def parse_resume(uploaded_file):
    data = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return extract_text_from_pdf(data)
    if name.endswith('.docx') or name.endswith('.doc'):
        return extract_text_from_docx(data)
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def extract_keywords_tfidf(text, top_n=10):
    if TfidfVectorizer is None:
        return []
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf = vec.fit_transform([text])
    indices = tfidf.toarray()[0].argsort()[::-1][:top_n]
    features = vec.get_feature_names_out()
    return [features[i] for i in indices]


def recommend_jobs(resume_text, jobs_df, top_n=5, use_transformer=False):
    if jobs_df.empty or not resume_text.strip():
        return jobs_df.head(0)
    if use_transformer and S2_AVAILABLE:
        corpus = jobs_df['description'].fillna('').tolist()
        emb = S2_MODEL.encode(corpus, convert_to_numpy=True)
        r_emb = S2_MODEL.encode([resume_text], convert_to_numpy=True)
        sims = cosine_similarity(r_emb, emb).flatten()
        jobs = jobs_df.copy()
        jobs['score'] = sims
        return jobs.sort_values('score', ascending=False).head(top_n)
    # fallback TF-IDF
    if TfidfVectorizer is None:
        return jobs_df.head(0)
    corpus = jobs_df['description'].fillna('')
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform(corpus.tolist() + [resume_text])
    resume_vec = tfidf[-1]
    job_vecs = tfidf[:-1]
    cosine_sim = linear_kernel(resume_vec, job_vecs).flatten()
    jobs = jobs_df.copy()
    jobs['score'] = cosine_sim
    return jobs.sort_values('score', ascending=False).head(top_n)


def resume_feedback(text):
    suggestions = []
    if not re.search(r'\b(이름|연락처|전화번호|email|이메일)\b', text, re.I):
        suggestions.append('연락처나 이메일 정보가 명시되어 있지 않습니다.')
    if not re.search(r'\b(경력|경력사항|직무)\b', text):
        suggestions.append('경력(직무) 섹션을 더 자세히 작성해보세요.')
    if not re.search(r'\b(학력|학교|졸업)\b', text):
        suggestions.append('학력 정보를 추가하면 좋습니다.')
    years = re.findall(r'(?:(19|20)\d{2})', text)
    if len(years) >= 2:
        suggestions.append('연도 표기가 산재해 있습니다. 공백 기간은 간단히 설명하세요.')
    common_skills = ['python','sql','excel','java','javascript','react','aws','docker']
    found = [s for s in common_skills if re.search(r'\b'+s+'\b', text, re.I)]
    if not found:
        suggestions.append('자주 요청되는 스킬(예: Python, SQL 등)을 명시해 보세요.')
    if not suggestions:
        suggestions.append('이력서가 전반적으로 양호해 보입니다. 세부 성과를 더 수치화해보세요.')
    return suggestions


def jobkorea_search(query, region=None, exp=None, pages=1):
    """
    시도형 간단 스크래퍼: 잡코리아 검색 페이지에서 공고 링크/타이틀을 수집합니다.
    실패 시 빈 데이터프레임 반환.
    """
    results = []
    if BeautifulSoup is None:
        return pd.DataFrame(results, columns=['company','title','region','salary','exp_level','description','url'])
    base = 'https://www.jobkorea.co.kr/Search/?stext='
    q = quote_plus(query)
    try:
        for p in range(1, pages+1):
            url = f'{base}{q}&Page_No={p}'
            r = requests.get(url, timeout=8, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, 'html.parser')
            # find links to recruitment pages
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/Recruit/' in href or '/Co/' in href:
                    title = a.get_text(strip=True)
                    link = requests.compat.urljoin('https://www.jobkorea.co.kr', href)
                    results.append({'company':'','title':title,'region':'','salary':None,'exp_level':'','description':'','url':link})
            if len(results) >= 50:
                break
    except Exception:
        return pd.DataFrame(results, columns=['company','title','region','salary','exp_level','description','url'])
    return pd.DataFrame(results, columns=['company','title','region','salary','exp_level','description','url'])


def send_email(smtp_server, smtp_port, username, password, to_email, subject, body, attachments=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = username
    msg['To'] = to_email
    msg.set_content(body)
    if attachments:
        for name, content, mime in attachments:
            msg.add_attachment(content, maintype=mime.split('/')[0], subtype=mime.split('/')[1], filename=name)
    try:
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return True, 'Sent'
    except Exception as e:
        return False, str(e)


def to_csv_download(df):
    return df.to_csv(index=False).encode('utf-8')


def main():
    local_css()
    jobs = load_jobs()

    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True

    with st.sidebar:
        st.markdown('<div class="big-logo">잡데렐라</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">당신의 동화 같은 커리어를 시작하는 곳, 잡데렐라!</div>', unsafe_allow_html=True)
        menu = st.radio('메뉴', ['이력서 업로드','추천 채용 공고','직무 인터뷰','합격자소서 예시','콘텐츠LAB','취업톡톡'])
        st.markdown('---')
        st.caption('실시간 잡코리아 연동 및 이메일 전송은 선택 기능입니다.')

    if st.session_state.first_visit:
        st.session_state.first_visit = False
        st.info('이력서를 업로드하면 잡데렐라가 맞춤형 추천과 정보를 제공합니다!')

    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="big-logo">🫧 잡데렐라</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">맞춤형 커리어 추천 서비스</div>', unsafe_allow_html=True)
    with col2:
        st.button('프로필 설정', disabled=True)

    if menu == '이력서 업로드':
        st.header('이력서 업로드')
        st.markdown('PDF 또는 Word(.docx) 파일을 업로드하면 분석합니다.')
        uploaded = st.file_uploader('이력서를 업로드하세요', type=['pdf','docx','doc'], accept_multiple_files=False)
        if uploaded is not None:
            text = parse_resume(uploaded)
            st.success('이력서가 업로드되어 분석되었습니다.')
            st.subheader('이력서 요약')
            st.text_area('원문', value=text[:5000], height=250)
            st.subheader('추출된 키워드')
            kws = extract_keywords_tfidf(text, top_n=12)
            st.write(', '.join(kws))
            st.subheader('개선 피드백')
            for s in resume_feedback(text):
                st.write('- ' + s)
            st.session_state['resume_text'] = text

    elif menu == '추천 채용 공고':
        st.header('추천 채용 공고')
        resume_text = st.session_state.get('resume_text','')
        st.sidebar.markdown('---')
        use_live = st.sidebar.checkbox('잡코리아 실시간 연동(시도)', value=False)
        use_transformer = st.sidebar.checkbox('고급 NLP(임베딩) 사용', value=S2_AVAILABLE)
        regions = ['전체'] + sorted(jobs['region'].dropna().unique().tolist())
        chosen_region = st.sidebar.selectbox('지역', regions)
        exp_filter = st.sidebar.selectbox('경력', ['전체','신입','경력'])
        salary_min = st.sidebar.number_input('최저 연봉(만원)', value=0, step=100)

        if use_live and resume_text:
            # build a simple query from top keywords
            kws = extract_keywords_tfidf(resume_text, top_n=5)
            query = ' '.join(kws) if kws else resume_text.split()[:5]
            live = jobkorea_search(query, pages=1)
            df_candidates = pd.concat([live, jobs], ignore_index=True, sort=False).drop_duplicates(subset=['title','url'], keep='first')
        else:
            df_candidates = jobs.copy()

        if resume_text:
            recs = recommend_jobs(resume_text, df_candidates, top_n=50, use_transformer=use_transformer)
        else:
            recs = df_candidates.copy()

        if chosen_region != '전체':
            recs = recs[recs['region']==chosen_region]
        if exp_filter != '전체':
            recs = recs[recs['exp_level'].str.contains(exp_filter, na=False)]
        if salary_min > 0:
            recs = recs[recs['salary'].fillna(0) >= salary_min]

        st.write(f'추천 공고: {len(recs)}개')
        if not recs.empty:
            display = recs[['company','title','region','salary']].reset_index(drop=True)
            st.dataframe(display)
            idx = st.number_input('상세보기: 공고 선택(번호)', min_value=0, max_value=max(0,len(display)-1), value=0)
            job = recs.reset_index(drop=True).iloc[int(idx)]
            st.subheader(job.get('title',''))
            st.write('회사:', job.get('company',''))
            st.write('지역:', job.get('region',''))
            st.write('연봉(만원):', job.get('salary',''))
            st.write('상세:', job.get('description',''))
            st.markdown(f'[공고로 이동]({job.get("url","#")})')

            if st.button('선택 공고 CSV로 저장'):
                csv = to_csv_download(pd.DataFrame([job]))
                st.download_button('다운로드', csv, file_name='job_export.csv', mime='text/csv')

            st.markdown('---')
            st.subheader('이메일로 전송(설정 필요)')
            smtp_server = st.text_input('SMTP 서버 (예: smtp.gmail.com)', value='')
            smtp_port = st.number_input('포트', value=465)
            smtp_user = st.text_input('보내는 이메일(계정)', value='')
            smtp_pass = st.text_input('비밀번호(앱 비밀번호 권장)', type='password')
            to_email = st.text_input('받는 이메일', value='')
            if st.button('이메일 전송(실제)'):
                if not all([smtp_server, smtp_port, smtp_user, smtp_pass, to_email]):
                    st.error('SMTP 설정과 받는 이메일을 모두 입력하세요.')
                else:
                    subject = f"[잡데렐라] 추천 공고: {job.get('title','')}"
                    body = f"회사: {job.get('company','')}\n지역: {job.get('region','')}\n연봉: {job.get('salary','')}\n\n상세: {job.get('description','')}\n링크: {job.get('url','')}"
                    ok, msg = send_email(smtp_server, int(smtp_port), smtp_user, smtp_pass, to_email, subject, body, attachments=None)
                    if ok:
                        st.success('이메일 전송 성공')
                    else:
                        st.error('이메일 전송 실패: ' + msg)
        else:
            st.info('조건에 맞는 공고가 없습니다. 필터를 조정해보세요.')

    elif menu == '직무 인터뷰':
        st.header('직무 인터뷰')
        st.markdown('업로드한 이력서와 연결된 직무의 인터뷰 팁과 모의 질문을 제공합니다.')
        resume_text = st.session_state.get('resume_text','')
        if not resume_text:
            st.info('먼저 이력서를 업로드하면 맞춤형 인터뷰 콘텐츠를 제공합니다.')
        else:
            kws = extract_keywords_tfidf(resume_text, top_n=8)
            st.subheader('추천 인터뷰 주제')
            for k in kws[:6]:
                st.write('- ' + k)
            st.subheader('모의 질문')
            st.write('1) 최근 담당한 프로젝트에서 가장 어려웠던 점은 무엇인가요?')
            st.write('2) 해당 직무에서 중요한 핵심 역량은 무엇이라고 생각하나요?')

    elif menu == '합격자소서 예시':
        st.header('합격자소서 예시')
        st.markdown('업로드 이력서의 직무에 맞춘 합격자소서 예시를 제공합니다.')
        st.write('예시 1: [지원동기 및 경험 기반 사례 설명] ...')

    elif menu == '콘텐츠LAB':
        st.header('콘텐츠LAB')
        st.write('직무별 공부자료, 추천 강의, 마인드셋 콘텐츠를 제공합니다.')
        st.write('- Python 기초 강의: 인프런/패스트캠퍼스 등')

    else:
        st.header('취업톡톡')
        st.write('커뮤니티 기반 질문과 답변 (예시 기능)')


if __name__ == '__main__':
    main()
