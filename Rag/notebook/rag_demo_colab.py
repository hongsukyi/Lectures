# ============================================================
# 2일차 오후 강사 데모: 미니 RAG 파이프라인
# 대상: 선수지식(Python·딥러닝·NLP) 보유 연구원
# 환경: Google Colab | 강사 API 키만 사용
# ============================================================

# ── 0. 패키지 설치 ──────────────────────────────────────────
!pip install -q openai langchain langchain-openai langchain-community faiss-cpu pypdf


# ── 1. API 키 설정 (Colab Secrets 사용 — 화면 노출 방지) ────
from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

# 로컬 실행 시 대체 방법:
# os.environ["OPENAI_API_KEY"] = "sk-..."


# ── 2. 데모용 샘플 문서 준비 ────────────────────────────────
# 강사 준비: 연구윤리 규정 PDF 또는 샘플 논문 txt
# 아래는 텍스트로 직접 입력하는 방식 (PDF 없을 때 대비)

sample_text = """
[연구데이터 관리 지침 - 샘플]

제1조 (목적)
이 지침은 연구데이터의 체계적 관리와 공유를 통해 연구의 투명성과
재현성을 높이고, 연구성과의 활용을 극대화하는 것을 목적으로 한다.

제2조 (적용 범위)
이 지침은 정부 출연 연구기관에서 수행하는 모든 연구과제에 적용한다.
단, 보안 등급이 부여된 연구과제는 별도 지침을 따른다.

제3조 (연구데이터의 정의)
연구데이터란 연구 수행 과정에서 생성·수집·처리·분석된 데이터로,
실험 결과, 관측값, 설문 응답, 인터뷰 기록 등을 포함한다.

제4조 (데이터 관리 계획)
연구책임자는 연구 시작 후 3개월 이내에 데이터 관리 계획(DMP)을
수립하고 기관 데이터 관리 시스템에 등록하여야 한다.

제5조 (데이터 보존 기간)
연구데이터는 연구 종료 후 최소 10년간 보존하여야 한다.
단, 개인정보가 포함된 데이터는 관련 법령에 따라 별도 관리한다.

제6조 (데이터 공개 원칙)
공공 재원으로 수행된 연구의 데이터는 원칙적으로 공개를 권장한다.
연구책임자는 데이터 공개 여부와 시기를 DMP에 명시하여야 한다.

제7조 (보안 및 접근 통제)
민감 연구데이터에 대해서는 접근 권한을 제한하고, 암호화 등
보안 조치를 적용하여야 한다.
"""

# 파일로 저장
with open("sample_research_policy.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)

print("샘플 문서 준비 완료")


# ── 3. 문서 로드 ─────────────────────────────────────────────
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# txt 로드
loader = TextLoader("sample_research_policy.txt", encoding="utf-8")
documents = loader.load()

# PDF 로드 시 대체 코드:
# loader = PyPDFLoader("your_document.pdf")
# documents = loader.load()

print(f"로드된 문서 수: {len(documents)}")
print(f"전체 텍스트 길이: {len(documents[0].page_content)} 자")


# ── 4. 청킹 ─────────────────────────────────────────────────
# ★ 시연 포인트 1: chunk_size를 바꿔가며 결과 비교
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 실험 A: 작은 청크 (정밀하지만 맥락이 잘릴 수 있음)
splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=100,       # ← 이 값을 바꿔 시연
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", ". ", " "]
)

# 실험 B: 큰 청크 (맥락 풍부하지만 노이즈 포함 가능)
splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ← 이 값을 바꿔 시연
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ". ", " "]
)

chunks_small = splitter_small.split_documents(documents)
chunks_large = splitter_large.split_documents(documents)

print(f"\n[청크 크기 100] 생성된 청크 수: {len(chunks_small)}")
print(f"[청크 크기 500] 생성된 청크 수: {len(chunks_large)}")

# 청크 내용 미리보기
print("\n── 작은 청크 첫 3개 ──")
for i, c in enumerate(chunks_small[:3]):
    print(f"[{i}] {c.page_content[:80]}...")

print("\n── 큰 청크 첫 2개 ──")
for i, c in enumerate(chunks_large[:2]):
    print(f"[{i}] {c.page_content[:120]}...")


# ── 5. 임베딩 + 벡터 저장소 ─────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 작은 청크로 벡터 저장소 구성
vectorstore_small = FAISS.from_documents(chunks_small, embeddings)
# 큰 청크로 벡터 저장소 구성
vectorstore_large = FAISS.from_documents(chunks_large, embeddings)

print("임베딩 완료 — 벡터 저장소 구성됨")


# ── 6. 검색 + 생성 ───────────────────────────────────────────
# ★ 시연 포인트 2: k값을 바꿔가며 답변 풍부도 비교
# ★ 시연 포인트 3: 시스템 프롬프트 변경으로 환각 차이 비교
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def run_rag(question, vectorstore, k=3, use_strict_prompt=True):
    """
    RAG 파이프라인 실행 함수
    - k: 참조할 청크 수 (시연 포인트 2)
    - use_strict_prompt: 문서 기반 엄격 응답 여부 (시연 포인트 3)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 일반 프롬프트
    general_template = """
다음 문맥을 참고하여 질문에 답하세요.

문맥:
{context}

질문: {question}
답변:"""

    # 엄격한 근거 기반 프롬프트 (환각 방지)
    strict_template = """
반드시 아래 제공된 문서 내용만을 근거로 답하세요.
문서에 없는 내용은 "제공된 문서에서 확인할 수 없습니다"라고 답하세요.
답변 끝에 참고한 문서 조항을 명시하세요.

문서 내용:
{context}

질문: {question}
답변 (근거 포함):"""

    template = strict_template if use_strict_prompt else general_template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True   # 참조 청크 반환
    )

    result = chain.invoke({"query": question})

    print(f"\n{'='*55}")
    print(f"질문: {question}")
    print(f"k={k} | 엄격 프롬프트={'ON' if use_strict_prompt else 'OFF'}")
    print(f"{'─'*55}")
    print(f"답변:\n{result['result']}")
    print(f"{'─'*55}")
    print(f"참조된 청크 ({len(result['source_documents'])}개):")
    for i, doc in enumerate(result['source_documents']):
        print(f"  [{i+1}] {doc.page_content[:80]}...")
    print(f"{'='*55}")
    return result


# ── 7. 시연 실행 ─────────────────────────────────────────────
question = "연구데이터 보존 기간은 얼마나 되나요?"

print("\n\n[시연 1] 청크 크기 비교")
print("── 작은 청크(100자)로 검색 ──")
run_rag(question, vectorstore_small, k=3)

print("\n── 큰 청크(500자)로 검색 ──")
run_rag(question, vectorstore_large, k=3)


print("\n\n[시연 2] k값 비교 (큰 청크 기준)")
print("── k=1 (청크 1개만 참조) ──")
run_rag(question, vectorstore_large, k=1)

print("\n── k=5 (청크 5개 참조) ──")
run_rag(question, vectorstore_large, k=5)


print("\n\n[시연 3] 프롬프트 비교")
q_hallucination = "연구데이터 공개 시 인센티브 제도가 있나요?"  # 문서에 없는 내용

print("── 일반 프롬프트 (환각 발생 가능) ──")
run_rag(q_hallucination, vectorstore_large, k=3, use_strict_prompt=False)

print("\n── 엄격 프롬프트 (근거 기반) ──")
run_rag(q_hallucination, vectorstore_large, k=3, use_strict_prompt=True)


# ── 8. (선택) 대화형 RAG — 수강생 질문 직접 입력 ─────────────
# 강사가 수강생에게 "질문 하나씩 받아서 실시간 실행" 용도
print("\n\n[선택 시연] 수강생 질문 직접 입력")
while True:
    q = input("\n질문 입력 (종료: q): ").strip()
    if q.lower() == "q":
        break
    run_rag(q, vectorstore_large, k=3, use_strict_prompt=True)
