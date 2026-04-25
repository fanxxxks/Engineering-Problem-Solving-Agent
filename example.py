print("正在加载本地知识库...")
db_load_path = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vectorstore = FAISS.load_local(db_load_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print("✅ FAISS 知识库加载成功！")
except Exception as e:
    print(f"❌ FAISS 知识库加载失败，请先运行 build.py。\n详细信息: {e}")
    exit()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PromptTemplate.from_template(
        "根据以下背景知识回答问题。\n\n背景知识:\n{context}\n\n问题: {question}\n\n答案:"
    )
    | llm
    | StrOutputParser()
)

# ---------- PDF 知识库 ----------
pdf_vectorstore = None
pdf_path = "mathandphyeqs.pdf"

if os.path.exists(pdf_path):
    print("正在加载 PDF 知识库...")
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        raw_pages = pdf_loader.load()
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_docs = pdf_splitter.split_documents(raw_pages)
        pdf_vectorstore = FAISS.from_documents(pdf_docs, embeddings)  # 复用同一 embeddings
        print(f"✅ PDF 知识库加载成功！共 {len(pdf_docs)} 个片段。")
    except Exception as e:
        print(f"⚠️  PDF 加载失败，将跳过 PDF 工具。\n详细信息: {e}")
else:
    print("⚠️  未找到 PDF 文件，将跳过 PDF 工具。")

# =======================================================
# 第三步：定义工具
# =======================================================

# 工具1：原有 FAISS 知识库检索
def retrieve_knowledge(query: str) -> str:
    return rag_chain.invoke(query)

# 工具2：PDF 原文检索
def pdf_quick_search(query: str) -> str:
    if pdf_vectorstore is None:
        return "PDF 知识库未加载，无法使用此工具。"
    results = pdf_vectorstore.similarity_search(query, k=3)
    return "\n---\n".join([res.page_content for res in results])

# 工具3：SymPy 数理计算引擎（替代原有 numexpr Calculator）
def physics_math_solver(query: str) -> str:
    """
    执行 Python 数理计算代码。
    支持 sympy（符号运算：求导、积分、方程求解）和 math/numpy（数值计算）。
    """
    safe_globals = {
        "sympy": __import__("sympy"),
        "math":  __import__("math"),
        "np":    __import__("numpy"),
        # 常用 sympy 函数直接暴露，方便模型调用
        "symbols":   __import__("sympy").symbols,
        "solve":     __import__("sympy").solve,
        "diff":      __import__("sympy").diff,
        "integrate": __import__("sympy").integrate,
        "simplify":  __import__("sympy").simplify,
        "pi":        __import__("sympy").pi,
        "sqrt":      __import__("sympy").sqrt,
        "print":     print,
    }

    old_stdout = sys.stdout
    sys.stdout = redirected = StringIO()
    try:
        exec(query, safe_globals)
        sys.stdout = old_stdout
        output = redirected.getvalue().strip()
        return output if output else "代码执行成功，但没有输出。请在代码中使用 print() 输出结果。"
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}"print("正在加载本地知识库...")
db_load_path = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vectorstore = FAISS.load_local(db_load_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print("✅ FAISS 知识库加载成功！")
except Exception as e:
    print(f"❌ FAISS 知识库加载失败，请先运行 build.py。\n详细信息: {e}")
    exit()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PromptTemplate.from_template(
        "根据以下背景知识回答问题。\n\n背景知识:\n{context}\n\n问题: {question}\n\n答案:"
    )
    | llm
    | StrOutputParser()
)

# ---------- PDF 知识库 ----------
pdf_vectorstore = None
pdf_path = "mathandphyeqs.pdf"

if os.path.exists(pdf_path):
    print("正在加载 PDF 知识库...")
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        raw_pages = pdf_loader.load()
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_docs = pdf_splitter.split_documents(raw_pages)
        pdf_vectorstore = FAISS.from_documents(pdf_docs, embeddings)  # 复用同一 embeddings
        print(f"✅ PDF 知识库加载成功！共 {len(pdf_docs)} 个片段。")
    except Exception as e:
        print(f"⚠️  PDF 加载失败，将跳过 PDF 工具。\n详细信息: {e}")
else:
    print("⚠️  未找到 PDF 文件，将跳过 PDF 工具。")

# =======================================================
# 第三步：定义工具
# =======================================================

# 工具1：原有 FAISS 知识库检索
def retrieve_knowledge(query: str) -> str:
    return rag_chain.invoke(query)

# 工具2：PDF 原文检索
def pdf_quick_search(query: str) -> str:
    if pdf_vectorstore is None:
        return "PDF 知识库未加载，无法使用此工具。"
    results = pdf_vectorstore.similarity_search(query, k=3)
    return "\n---\n".join([res.page_content for res in results])

# 工具3：SymPy 数理计算引擎（替代原有 numexpr Calculator）
def physics_math_solver(query: str) -> str:
    """
    执行 Python 数理计算代码。
    支持 sympy（符号运算：求导、积分、方程求解）和 math/numpy（数值计算）。
    """
    safe_globals = {
        "sympy": __import__("sympy"),
        "math":  __import__("math"),
        "np":    __import__("numpy"),
        # 常用 sympy 函数直接暴露，方便模型调用
        "symbols":   __import__("sympy").symbols,
        "solve":     __import__("sympy").solve,
        "diff":      __import__("sympy").diff,
        "integrate": __import__("sympy").integrate,
        "simplify":  __import__("sympy").simplify,
        "pi":        __import__("sympy").pi,
        "sqrt":      __import__("sympy").sqrt,
        "print":     print,
    }

    old_stdout = sys.stdout
    sys.stdout = redirected = StringIO()
    try:
        exec(query, safe_globals)
        sys.stdout = old_stdout
        output = redirected.getvalue().strip()
        return output if output else "代码执行成功，但没有输出。请在代码中使用 print() 输出结果。"
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}"