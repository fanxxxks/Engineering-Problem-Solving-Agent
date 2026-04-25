---
jupyter:
  kernelspec:
    display_name: WYCity
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.13.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#790505b9 .cell .markdown}
# 第三届未央城赛道一的简单demo！

欢迎大家参加第三届未央城赛事🥳，这是一个简单的入门教程\
在这个教程中我会带着你用langchain和ReAct框架搭建一个简单的数理智能体，仅作为抛砖引玉，期待大家的发挥！

## 一、准备事项

### 一、大模型的准备

让我们回归最基本的问题：\
1.LangChain 是什么？\
简单来说，LangChain
是一个用来"粘合"大模型（LLM）和各种外部工具的开发框架（SDK）。\
如果没有 LangChain，直接调用大模型（比如
GPT-4）就像是你在和一个"只有大脑，没有手脚"的天才对话。他很聪明，但他记不住你们之前的聊天（没有内存），他也不能上网、不能翻看你电脑里的
Excel 表格（没有工具）。\
**LangChain 的作用就是给这个大脑装上"手脚"和"外挂"：**\
Chains（链）：把动作串联起来。比如：先读取用户的问题 -\> 翻译成英文 -\>
发给大模型 -\> 把答案翻译回中文。\
Agents（智能体）：让大模型自己决定该干什么。比如用户问"进动和章动分别是什么？"，LangChain
会让大模型思考，然后决定去调用"物理知识接口"，而不是直接瞎编。\
Retrieval（检索/RAG）：给大模型外挂知识库。比如让它读取shuo的基物1教材，然后回答相关问题。\
Memory（记忆）：让大模型记住上下文，能进行多轮对话。

### 一句话总结：LangChain 是大模型应用开发的脚手架，它帮你把繁琐的代码封装好了，让你能快速搭出一个 AI 应用。

2.ReAct范式是什么？\
ReAct 是 Reasoning（推理） + Acting（行动） 的缩写。它的流程是：\
Thinking（思考）：大模型看用户的问题，心想："用户问的是数理问题，我应该先去查公式。"\
Action（行动）：大模型输出指令，调用你写的"数理知识库工具"。\
Observation（观察）：工具返回了结果（比如 F=ma）。\
Thinking（再思考）：大模型心想："公式有了，现在我需要计算具体的数值。"\
Action（再行动）：大模型调用"计算器工具"。\
Final Answer（最终回答）：大模型输出最终结果。\
**从这里我们可以得出结论：我们首先需要一个大模型的API KEY**

一个方便获取的地方： <https://easycompute.cs.tsinghua.edu.cn/login>\
在这里申请获取大模型KEY后就可以使用了，例如：
:::

::: {#535f803d .cell .code execution_count="2"}
``` python
! pip install langchain_openai
```

::: {.output .stream .stdout}
    Collecting langchain_openai
      Downloading langchain_openai-1.1.11-py3-none-any.whl.metadata (3.1 kB)
    Collecting langchain-core<2.0.0,>=1.2.18 (from langchain_openai)
      Downloading langchain_core-1.2.19-py3-none-any.whl.metadata (4.4 kB)
    Collecting openai<3.0.0,>=2.26.0 (from langchain_openai)
      Downloading openai-2.28.0-py3-none-any.whl.metadata (29 kB)
    Collecting tiktoken<1.0.0,>=0.7.0 (from langchain_openai)
      Downloading tiktoken-0.12.0-cp313-cp313-win_amd64.whl.metadata (6.9 kB)
    Collecting jsonpatch<2.0.0,>=1.33.0 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading jsonpatch-1.33-py2.py3-none-any.whl.metadata (3.0 kB)
    Collecting langsmith<1.0.0,>=0.3.45 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading langsmith-0.7.17-py3-none-any.whl.metadata (15 kB)
    Requirement already satisfied: packaging>=23.2.0 in D:\miniconda3\envs\WYCity\Lib\site-packages (from langchain-core<2.0.0,>=1.2.18->langchain_openai) (25.0)
    Collecting pydantic<3.0.0,>=2.7.4 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
    Collecting pyyaml<7.0.0,>=5.3.0 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading pyyaml-6.0.3-cp313-cp313-win_amd64.whl.metadata (2.4 kB)
    Collecting tenacity!=8.4.0,<10.0.0,>=8.1.0 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading tenacity-9.1.4-py3-none-any.whl.metadata (1.2 kB)
    Collecting typing-extensions<5.0.0,>=4.7.0 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
    Collecting uuid-utils<1.0,>=0.12.0 (from langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading uuid_utils-0.14.1-cp39-abi3-win_amd64.whl.metadata (4.9 kB)
    Collecting jsonpointer>=1.9 (from jsonpatch<2.0.0,>=1.33.0->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading jsonpointer-3.0.0-py2.py3-none-any.whl.metadata (2.3 kB)
    Collecting httpx<1,>=0.23.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
    Collecting orjson>=3.9.14 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading orjson-3.11.7-cp313-cp313-win_amd64.whl.metadata (43 kB)
    Collecting requests-toolbelt>=1.0.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl.metadata (14 kB)
    Collecting requests>=2.0.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
    Collecting xxhash>=3.0.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading xxhash-3.6.0-cp313-cp313-win_amd64.whl.metadata (13 kB)
    Collecting zstandard>=0.23.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading zstandard-0.25.0-cp313-cp313-win_amd64.whl.metadata (3.3 kB)
    Collecting anyio (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading anyio-4.12.1-py3-none-any.whl.metadata (4.3 kB)
    Collecting certifi (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading certifi-2026.2.25-py3-none-any.whl.metadata (2.5 kB)
    Collecting httpcore==1.* (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
    Collecting idna (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
    Collecting h11>=0.16 (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
    Collecting distro<2,>=1.7.0 (from openai<3.0.0,>=2.26.0->langchain_openai)
      Downloading distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)
    Collecting jiter<1,>=0.10.0 (from openai<3.0.0,>=2.26.0->langchain_openai)
      Downloading jiter-0.13.0-cp313-cp313-win_amd64.whl.metadata (5.3 kB)
    Collecting sniffio (from openai<3.0.0,>=2.26.0->langchain_openai)
      Downloading sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
    Collecting tqdm>4 (from openai<3.0.0,>=2.26.0->langchain_openai)
      Downloading tqdm-4.67.3-py3-none-any.whl.metadata (57 kB)
    Collecting annotated-types>=0.6.0 (from pydantic<3.0.0,>=2.7.4->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
    Collecting pydantic-core==2.41.5 (from pydantic<3.0.0,>=2.7.4->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading pydantic_core-2.41.5-cp313-cp313-win_amd64.whl.metadata (7.4 kB)
    Collecting typing-inspection>=0.4.2 (from pydantic<3.0.0,>=2.7.4->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
    Collecting regex>=2022.1.18 (from tiktoken<1.0.0,>=0.7.0->langchain_openai)
      Downloading regex-2026.2.28-cp313-cp313-win_amd64.whl.metadata (41 kB)
    Collecting charset_normalizer<4,>=2 (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Downloading charset_normalizer-3.4.6-cp313-cp313-win_amd64.whl.metadata (41 kB)
    Collecting urllib3<3,>=1.21.1 (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.18->langchain_openai)
      Using cached urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
    Requirement already satisfied: colorama in D:\miniconda3\envs\WYCity\Lib\site-packages (from tqdm>4->openai<3.0.0,>=2.26.0->langchain_openai) (0.4.6)
    Downloading langchain_openai-1.1.11-py3-none-any.whl (87 kB)
    Downloading langchain_core-1.2.19-py3-none-any.whl (503 kB)
    Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
    Downloading langsmith-0.7.17-py3-none-any.whl (359 kB)
    Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
    Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
    Downloading openai-2.28.0-py3-none-any.whl (1.1 MB)
       ---------------------------------------- 0.0/1.1 MB ? eta -:--:--
       ---------------------------------------- 1.1/1.1 MB 14.4 MB/s  0:00:00
    Downloading anyio-4.12.1-py3-none-any.whl (113 kB)
    Downloading distro-1.9.0-py3-none-any.whl (20 kB)
    Downloading jiter-0.13.0-cp313-cp313-win_amd64.whl (202 kB)
    Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
    Downloading pydantic_core-2.41.5-cp313-cp313-win_amd64.whl (2.0 MB)
       ---------------------------------------- 0.0/2.0 MB ? eta -:--:--
       ---------------------------------------- 2.0/2.0 MB 17.5 MB/s  0:00:00
    Downloading pyyaml-6.0.3-cp313-cp313-win_amd64.whl (154 kB)
    Downloading tenacity-9.1.4-py3-none-any.whl (28 kB)
    Downloading tiktoken-0.12.0-cp313-cp313-win_amd64.whl (879 kB)
       ---------------------------------------- 0.0/879.1 kB ? eta -:--:--
       ---------------------------------------- 879.1/879.1 kB 15.0 MB/s  0:00:00
    Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
    Downloading uuid_utils-0.14.1-cp39-abi3-win_amd64.whl (187 kB)
    Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
    Downloading h11-0.16.0-py3-none-any.whl (37 kB)
    Using cached idna-3.11-py3-none-any.whl (71 kB)
    Downloading jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)
    Downloading orjson-3.11.7-cp313-cp313-win_amd64.whl (124 kB)
    Downloading regex-2026.2.28-cp313-cp313-win_amd64.whl (277 kB)
    Using cached requests-2.32.5-py3-none-any.whl (64 kB)
    Downloading charset_normalizer-3.4.6-cp313-cp313-win_amd64.whl (154 kB)
    Using cached urllib3-2.6.3-py3-none-any.whl (131 kB)
    Downloading certifi-2026.2.25-py3-none-any.whl (153 kB)
    Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
    Downloading tqdm-4.67.3-py3-none-any.whl (78 kB)
    Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
    Downloading xxhash-3.6.0-cp313-cp313-win_amd64.whl (31 kB)
    Downloading zstandard-0.25.0-cp313-cp313-win_amd64.whl (506 kB)
    Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)
    Installing collected packages: zstandard, xxhash, uuid-utils, urllib3, typing-extensions, tqdm, tenacity, sniffio, regex, pyyaml, orjson, jsonpointer, jiter, idna, h11, distro, charset_normalizer, certifi, annotated-types, typing-inspection, requests, pydantic-core, jsonpatch, httpcore, anyio, tiktoken, requests-toolbelt, pydantic, httpx, openai, langsmith, langchain-core, langchain_openai

       - --------------------------------------  1/33 [xxhash]
       --- ------------------------------------  3/33 [urllib3]
       --- ------------------------------------  3/33 [urllib3]
       --- ------------------------------------  3/33 [urllib3]
       --- ------------------------------------  3/33 [urllib3]
       --- ------------------------------------  3/33 [urllib3]
       --- ------------------------------------  3/33 [urllib3]
       ---- -----------------------------------  4/33 [typing-extensions]
       ------ ---------------------------------  5/33 [tqdm]
       ------ ---------------------------------  5/33 [tqdm]
       ------ ---------------------------------  5/33 [tqdm]
       ------ ---------------------------------  5/33 [tqdm]
       ------ ---------------------------------  5/33 [tqdm]
       ------ ---------------------------------  5/33 [tqdm]
       ------- --------------------------------  6/33 [tenacity]
       ------- --------------------------------  6/33 [tenacity]
       --------- ------------------------------  8/33 [regex]
       --------- ------------------------------  8/33 [regex]
       ---------- -----------------------------  9/33 [pyyaml]
       ---------- -----------------------------  9/33 [pyyaml]
       ---------- -----------------------------  9/33 [pyyaml]
       ------------- -------------------------- 11/33 [jsonpointer]
       --------------- ------------------------ 13/33 [idna]
       --------------- ------------------------ 13/33 [idna]
       ---------------- ----------------------- 14/33 [h11]
       ---------------- ----------------------- 14/33 [h11]
       ---------------- ----------------------- 14/33 [h11]
       ------------------ --------------------- 15/33 [distro]
       ------------------- -------------------- 16/33 [charset_normalizer]
       ------------------- -------------------- 16/33 [charset_normalizer]
       ------------------- -------------------- 16/33 [charset_normalizer]
       ------------------- -------------------- 16/33 [charset_normalizer]
       ----------------------- ---------------- 19/33 [typing-inspection]
       ------------------------ --------------- 20/33 [requests]
       ------------------------ --------------- 20/33 [requests]
       ------------------------ --------------- 20/33 [requests]
       ------------------------- -------------- 21/33 [pydantic-core]
       -------------------------- ------------- 22/33 [jsonpatch]
       --------------------------- ------------ 23/33 [httpcore]
       --------------------------- ------------ 23/33 [httpcore]
       --------------------------- ------------ 23/33 [httpcore]
       --------------------------- ------------ 23/33 [httpcore]
       --------------------------- ------------ 23/33 [httpcore]
       ----------------------------- ---------- 24/33 [anyio]
       ----------------------------- ---------- 24/33 [anyio]
       ----------------------------- ---------- 24/33 [anyio]
       ----------------------------- ---------- 24/33 [anyio]
       ----------------------------- ---------- 24/33 [anyio]
       ----------------------------- ---------- 24/33 [anyio]
       ------------------------------ --------- 25/33 [tiktoken]
       ------------------------------- -------- 26/33 [requests-toolbelt]
       ------------------------------- -------- 26/33 [requests-toolbelt]
       ------------------------------- -------- 26/33 [requests-toolbelt]
       ------------------------------- -------- 26/33 [requests-toolbelt]
       ------------------------------- -------- 26/33 [requests-toolbelt]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       -------------------------------- ------- 27/33 [pydantic]
       --------------------------------- ------ 28/33 [httpx]
       --------------------------------- ------ 28/33 [httpx]
       --------------------------------- ------ 28/33 [httpx]
       --------------------------------- ------ 28/33 [httpx]
       --------------------------------- ------ 28/33 [httpx]
       --------------------------------- ------ 28/33 [httpx]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ----------------------------------- ---- 29/33 [openai]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------ --- 30/33 [langsmith]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       ------------------------------------- -- 31/33 [langchain-core]
       -------------------------------------- - 32/33 [langchain_openai]
       -------------------------------------- - 32/33 [langchain_openai]
       -------------------------------------- - 32/33 [langchain_openai]
       ---------------------------------------- 33/33 [langchain_openai]

    Successfully installed annotated-types-0.7.0 anyio-4.12.1 certifi-2026.2.25 charset_normalizer-3.4.6 distro-1.9.0 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.11 jiter-0.13.0 jsonpatch-1.33 jsonpointer-3.0.0 langchain-core-1.2.19 langchain_openai-1.1.11 langsmith-0.7.17 openai-2.28.0 orjson-3.11.7 pydantic-2.12.5 pydantic-core-2.41.5 pyyaml-6.0.3 regex-2026.2.28 requests-2.32.5 requests-toolbelt-1.0.0 sniffio-1.3.1 tenacity-9.1.4 tiktoken-0.12.0 tqdm-4.67.3 typing-extensions-4.15.0 typing-inspection-0.4.2 urllib3-2.6.3 uuid-utils-0.14.1 xxhash-3.6.0 zstandard-0.25.0
:::
:::

::: {#8ba0909a .cell .code execution_count="6"}
``` python
from langchain_openai import ChatOpenAI
from pathlib import Path

api_key_file = Path("api_key.txt")
if not api_key_file.exists():
    raise FileNotFoundError("未找到 api_key.txt，请在项目根目录创建该文件并写入你的 API Key。")

API_KEY = api_key_file.read_text(encoding="utf-8").strip()
if not API_KEY:
    raise ValueError("api_key.txt 为空，请写入有效的 API Key。")

llm = ChatOpenAI(
    model="DeepSeek-V3.2",
    api_key=API_KEY,
    base_url="https://llmapi.paratera.com",
    temperature=0,
    streaming=True
)
```
:::

::: {#c15b9d30 .cell .markdown}
当然，大模型调试也有讲究，建议选思考时间不太长的大模型来调试，如果选deepseek-R1这种的思考类大模型你有福享了......\
不过放心！最后的大模型是统一的，我们不会在这里创造不公平因素！

### 2.langchain的配置 {#2langchain的配置}

这个地方参加过第二届未央城的同学肯定很熟悉，首先我们要创造虚拟环境，其次将相关的模块导入来用\
创建虚拟环境（以**cmd**中为例，分别为切换到D盘，切换到指定文件夹，创建虚拟环境，激活（进入）虚拟环境，安装相应依赖）\
指定版本号是为了让这些模块配套，防止你的程序之后找不到相关模块中的函数
:::

::: {#f981cb53 .cell .code vscode="{\"languageId\":\"shellscript\"}"}
``` python
D:
cd weyoungcity_3
python -m venv venv
.\venv\Scripts\activate
pip install langchain==0.3.25 langchain-core==0.3.65 langchain-community==0.3.24 langchain-openai==0.3.16
pip install numexpr sentence-transformers faiss-cpu
```
:::

::: {#f152e644 .cell .code execution_count="8" vscode="{\"languageId\":\"shellscript\"}"}
``` python
pip install numexpr sentence-transformers faiss-cpu
```

::: {.output .stream .stdout}
    Collecting numexpr
      Downloading numexpr-2.14.1-cp313-cp313-win_amd64.whl.metadata (9.3 kB)
    Collecting sentence-transformers
      Downloading sentence_transformers-5.3.0-py3-none-any.whl.metadata (16 kB)
    Collecting faiss-cpu
      Downloading faiss_cpu-1.13.2-cp313-cp313-win_amd64.whl.metadata (7.6 kB)
    Requirement already satisfied: numpy>=1.23.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from numexpr) (2.4.3)
    Collecting transformers<6.0.0,>=4.41.0 (from sentence-transformers)
      Downloading transformers-5.3.0-py3-none-any.whl.metadata (32 kB)
    Collecting huggingface-hub>=0.20.0 (from sentence-transformers)
      Downloading huggingface_hub-1.7.1-py3-none-any.whl.metadata (13 kB)
    Collecting torch>=1.11.0 (from sentence-transformers)
      Downloading torch-2.10.0-cp313-cp313-win_amd64.whl.metadata (31 kB)
    Collecting scikit-learn (from sentence-transformers)
      Using cached scikit_learn-1.8.0-cp313-cp313-win_amd64.whl.metadata (11 kB)
    Collecting scipy (from sentence-transformers)
      Downloading scipy-1.17.1-cp313-cp313-win_amd64.whl.metadata (60 kB)
    Requirement already satisfied: typing_extensions>=4.5.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from sentence-transformers) (4.15.0)
    Requirement already satisfied: tqdm in d:\miniconda3\envs\WYCity\Lib\site-packages (from sentence-transformers) (4.67.3)
    Requirement already satisfied: packaging>=20.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in d:\miniconda3\envs\WYCity\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (6.0.3)
    Requirement already satisfied: regex!=2019.12.17 in d:\miniconda3\envs\WYCity\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2026.2.28)
    Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading tokenizers-0.22.2-cp39-abi3-win_amd64.whl.metadata (7.4 kB)
    Collecting typer (from transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading typer-0.24.1-py3-none-any.whl.metadata (16 kB)
    Collecting safetensors>=0.4.3 (from transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading safetensors-0.7.0-cp38-abi3-win_amd64.whl.metadata (4.2 kB)
    Collecting filelock>=3.10.0 (from huggingface-hub>=0.20.0->sentence-transformers)
      Downloading filelock-3.25.2-py3-none-any.whl.metadata (2.0 kB)
    Collecting fsspec>=2023.5.0 (from huggingface-hub>=0.20.0->sentence-transformers)
      Downloading fsspec-2026.2.0-py3-none-any.whl.metadata (10 kB)
    Collecting hf-xet<2.0.0,>=1.4.2 (from huggingface-hub>=0.20.0->sentence-transformers)
      Downloading hf_xet-1.4.2-cp37-abi3-win_amd64.whl.metadata (4.9 kB)
    Requirement already satisfied: httpx<1,>=0.23.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (0.28.1)
    Requirement already satisfied: anyio in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (4.12.1)
    Requirement already satisfied: certifi in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (2026.2.25)
    Requirement already satisfied: httpcore==1.* in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (1.0.9)
    Requirement already satisfied: idna in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (3.11)
    Requirement already satisfied: h11>=0.16 in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (0.16.0)
    Collecting sympy>=1.13.3 (from torch>=1.11.0->sentence-transformers)
      Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
    Collecting networkx>=2.5.1 (from torch>=1.11.0->sentence-transformers)
      Using cached networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
    Collecting jinja2 (from torch>=1.11.0->sentence-transformers)
      Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
    Requirement already satisfied: setuptools in d:\miniconda3\envs\WYCity\Lib\site-packages (from torch>=1.11.0->sentence-transformers) (80.10.2)
    Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers)
      Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
    Requirement already satisfied: colorama in d:\miniconda3\envs\WYCity\Lib\site-packages (from tqdm->sentence-transformers) (0.4.6)
    Collecting MarkupSafe>=2.0 (from jinja2->torch>=1.11.0->sentence-transformers)
      Using cached markupsafe-3.0.3-cp313-cp313-win_amd64.whl.metadata (2.8 kB)
    Collecting joblib>=1.3.0 (from scikit-learn->sentence-transformers)
      Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
    Collecting threadpoolctl>=3.2.0 (from scikit-learn->sentence-transformers)
      Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
    Collecting click>=8.2.1 (from typer->transformers<6.0.0,>=4.41.0->sentence-transformers)
      Using cached click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
    Collecting shellingham>=1.3.0 (from typer->transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
    Collecting rich>=12.3.0 (from typer->transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading rich-14.3.3-py3-none-any.whl.metadata (18 kB)
    Collecting annotated-doc>=0.0.2 (from typer->transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)
    Collecting markdown-it-py>=2.2.0 (from rich>=12.3.0->typer->transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from rich>=12.3.0->typer->transformers<6.0.0,>=4.41.0->sentence-transformers) (2.19.2)
    Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->transformers<6.0.0,>=4.41.0->sentence-transformers)
      Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
    Downloading numexpr-2.14.1-cp313-cp313-win_amd64.whl (160 kB)
    Downloading sentence_transformers-5.3.0-py3-none-any.whl (512 kB)
    Downloading transformers-5.3.0-py3-none-any.whl (10.7 MB)
       ---------------------------------------- 0.0/10.7 MB ? eta -:--:--
       ----------- ---------------------------- 3.1/10.7 MB 16.2 MB/s eta 0:00:01
       --------------------- ------------------ 5.8/10.7 MB 14.3 MB/s eta 0:00:01
       -------------------------------- ------- 8.7/10.7 MB 14.2 MB/s eta 0:00:01
       -------------------------------------- - 10.2/10.7 MB 13.6 MB/s eta 0:00:01
       ---------------------------------------- 10.7/10.7 MB 11.0 MB/s  0:00:00
    Downloading huggingface_hub-1.7.1-py3-none-any.whl (616 kB)
       ---------------------------------------- 0.0/616.3 kB ? eta -:--:--
       ---------------------------------------- 616.3/616.3 kB 9.6 MB/s  0:00:00
    Downloading hf_xet-1.4.2-cp37-abi3-win_amd64.whl (3.7 MB)
       ---------------------------------------- 0.0/3.7 MB ? eta -:--:--
       -------------- ------------------------- 1.3/3.7 MB 8.0 MB/s eta 0:00:01
       ------------------------- -------------- 2.4/3.7 MB 7.1 MB/s eta 0:00:01
       ------------------------------- -------- 2.9/3.7 MB 5.0 MB/s eta 0:00:01
       ---------------------------------------- 3.7/3.7 MB 4.5 MB/s  0:00:00
    Downloading tokenizers-0.22.2-cp39-abi3-win_amd64.whl (2.7 MB)
       ---------------------------------------- 0.0/2.7 MB ? eta -:--:--
       ---------------------- ----------------- 1.6/2.7 MB 7.3 MB/s eta 0:00:01
       -------------------------------------- - 2.6/2.7 MB 6.1 MB/s eta 0:00:01
       ---------------------------------------- 2.7/2.7 MB 6.2 MB/s  0:00:00
    Downloading faiss_cpu-1.13.2-cp313-cp313-win_amd64.whl (18.9 MB)
       ---------------------------------------- 0.0/18.9 MB ? eta -:--:--
       - -------------------------------------- 0.8/18.9 MB 6.9 MB/s eta 0:00:03
       ---- ----------------------------------- 2.1/18.9 MB 4.9 MB/s eta 0:00:04
       ------ --------------------------------- 2.9/18.9 MB 4.5 MB/s eta 0:00:04
       ------- -------------------------------- 3.7/18.9 MB 4.3 MB/s eta 0:00:04
       --------- ------------------------------ 4.5/18.9 MB 4.2 MB/s eta 0:00:04
       ---------- ----------------------------- 5.0/18.9 MB 4.1 MB/s eta 0:00:04
       ----------- ---------------------------- 5.2/18.9 MB 3.8 MB/s eta 0:00:04
       ------------- -------------------------- 6.3/18.9 MB 3.6 MB/s eta 0:00:04
       -------------- ------------------------- 7.1/18.9 MB 3.7 MB/s eta 0:00:04
       ---------------- ----------------------- 7.9/18.9 MB 3.7 MB/s eta 0:00:03
       ------------------ --------------------- 8.7/18.9 MB 3.8 MB/s eta 0:00:03
       -------------------- ------------------- 9.7/18.9 MB 3.8 MB/s eta 0:00:03
       ---------------------- ----------------- 10.5/18.9 MB 3.8 MB/s eta 0:00:03
       ------------------------ --------------- 11.5/18.9 MB 3.9 MB/s eta 0:00:02
       -------------------------- ------------- 12.6/18.9 MB 3.9 MB/s eta 0:00:02
       --------------------------- ------------ 13.1/18.9 MB 3.8 MB/s eta 0:00:02
       ----------------------------- ---------- 13.9/18.9 MB 3.8 MB/s eta 0:00:02
       ------------------------------- -------- 14.7/18.9 MB 3.8 MB/s eta 0:00:02
       --------------------------------- ------ 15.7/18.9 MB 3.9 MB/s eta 0:00:01
       ---------------------------------- ----- 16.5/18.9 MB 3.9 MB/s eta 0:00:01
       ------------------------------------- -- 17.6/18.9 MB 3.9 MB/s eta 0:00:01
       ---------------------------------------  18.6/18.9 MB 4.0 MB/s eta 0:00:01
       ---------------------------------------- 18.9/18.9 MB 3.9 MB/s  0:00:04
    Downloading filelock-3.25.2-py3-none-any.whl (26 kB)
    Downloading fsspec-2026.2.0-py3-none-any.whl (202 kB)
    Downloading safetensors-0.7.0-cp38-abi3-win_amd64.whl (341 kB)
    Downloading torch-2.10.0-cp313-cp313-win_amd64.whl (113.8 MB)
       ---------------------------------------- 0.0/113.8 MB ? eta -:--:--
       ---------------------------------------- 0.8/113.8 MB 4.5 MB/s eta 0:00:26
        --------------------------------------- 1.8/113.8 MB 4.5 MB/s eta 0:00:25
        --------------------------------------- 2.6/113.8 MB 4.6 MB/s eta 0:00:25
       - -------------------------------------- 3.4/113.8 MB 4.4 MB/s eta 0:00:26
       - -------------------------------------- 4.2/113.8 MB 4.3 MB/s eta 0:00:26
       - -------------------------------------- 5.2/113.8 MB 4.3 MB/s eta 0:00:26
       -- ------------------------------------- 6.3/113.8 MB 4.4 MB/s eta 0:00:25
       -- ------------------------------------- 7.3/113.8 MB 4.4 MB/s eta 0:00:24
       -- ------------------------------------- 8.1/113.8 MB 4.5 MB/s eta 0:00:24
       --- ------------------------------------ 9.2/113.8 MB 4.5 MB/s eta 0:00:24
       --- ------------------------------------ 10.5/113.8 MB 4.6 MB/s eta 0:00:23
       ---- ----------------------------------- 11.5/113.8 MB 4.7 MB/s eta 0:00:22
       ---- ----------------------------------- 12.6/113.8 MB 4.7 MB/s eta 0:00:22
       ---- ----------------------------------- 13.6/113.8 MB 4.8 MB/s eta 0:00:22
       ----- ---------------------------------- 14.9/113.8 MB 4.8 MB/s eta 0:00:21
       ----- ---------------------------------- 16.3/113.8 MB 4.9 MB/s eta 0:00:20
       ------ --------------------------------- 17.3/113.8 MB 4.9 MB/s eta 0:00:20
       ------ --------------------------------- 18.6/113.8 MB 5.0 MB/s eta 0:00:20
       ------- -------------------------------- 19.9/113.8 MB 5.1 MB/s eta 0:00:19
       ------- -------------------------------- 21.2/113.8 MB 5.1 MB/s eta 0:00:19
       ------- -------------------------------- 22.5/113.8 MB 5.2 MB/s eta 0:00:18
       -------- ------------------------------- 23.9/113.8 MB 5.2 MB/s eta 0:00:18
       -------- ------------------------------- 25.2/113.8 MB 5.3 MB/s eta 0:00:17
       --------- ------------------------------ 26.5/113.8 MB 5.3 MB/s eta 0:00:17
       --------- ------------------------------ 28.0/113.8 MB 5.4 MB/s eta 0:00:17
       ---------- ----------------------------- 29.4/113.8 MB 5.4 MB/s eta 0:00:16
       ---------- ----------------------------- 30.4/113.8 MB 5.4 MB/s eta 0:00:16
       ----------- ---------------------------- 31.5/113.8 MB 5.4 MB/s eta 0:00:16
       ----------- ---------------------------- 32.0/113.8 MB 5.3 MB/s eta 0:00:16
       ----------- ---------------------------- 32.8/113.8 MB 5.3 MB/s eta 0:00:16
       ----------- ---------------------------- 33.6/113.8 MB 5.2 MB/s eta 0:00:16
       ------------ --------------------------- 34.3/113.8 MB 5.2 MB/s eta 0:00:16
       ------------ --------------------------- 35.1/113.8 MB 5.1 MB/s eta 0:00:16
       ------------ --------------------------- 35.9/113.8 MB 5.1 MB/s eta 0:00:16
       ------------ --------------------------- 36.7/113.8 MB 5.0 MB/s eta 0:00:16
       ------------- -------------------------- 37.5/113.8 MB 5.0 MB/s eta 0:00:16
       ------------- -------------------------- 38.3/113.8 MB 5.0 MB/s eta 0:00:16
       ------------- -------------------------- 39.1/113.8 MB 4.9 MB/s eta 0:00:16
       -------------- ------------------------- 39.8/113.8 MB 4.9 MB/s eta 0:00:16
       -------------- ------------------------- 40.6/113.8 MB 4.9 MB/s eta 0:00:16
       -------------- ------------------------- 41.4/113.8 MB 4.9 MB/s eta 0:00:15
       -------------- ------------------------- 42.5/113.8 MB 4.8 MB/s eta 0:00:15
       --------------- ------------------------ 43.3/113.8 MB 4.8 MB/s eta 0:00:15
       --------------- ------------------------ 44.3/113.8 MB 4.8 MB/s eta 0:00:15
       --------------- ------------------------ 45.4/113.8 MB 4.8 MB/s eta 0:00:15
       ---------------- ----------------------- 46.1/113.8 MB 4.8 MB/s eta 0:00:15
       ---------------- ----------------------- 46.9/113.8 MB 4.8 MB/s eta 0:00:14
       ---------------- ----------------------- 48.0/113.8 MB 4.8 MB/s eta 0:00:14
       ----------------- ---------------------- 49.0/113.8 MB 4.8 MB/s eta 0:00:14
       ----------------- ---------------------- 50.1/113.8 MB 4.8 MB/s eta 0:00:14
       ----------------- ---------------------- 51.1/113.8 MB 4.8 MB/s eta 0:00:14
       ------------------ --------------------- 52.2/113.8 MB 4.8 MB/s eta 0:00:13
       ------------------ --------------------- 53.5/113.8 MB 4.8 MB/s eta 0:00:13
       ------------------- -------------------- 54.5/113.8 MB 4.8 MB/s eta 0:00:13
       ------------------- -------------------- 55.6/113.8 MB 4.8 MB/s eta 0:00:13
       ------------------- -------------------- 56.4/113.8 MB 4.8 MB/s eta 0:00:12
       -------------------- ------------------- 57.4/113.8 MB 4.8 MB/s eta 0:00:12
       -------------------- ------------------- 58.5/113.8 MB 4.8 MB/s eta 0:00:12
       --------------------- ------------------ 59.8/113.8 MB 4.9 MB/s eta 0:00:12
       --------------------- ------------------ 60.8/113.8 MB 4.9 MB/s eta 0:00:11
       --------------------- ------------------ 62.1/113.8 MB 4.9 MB/s eta 0:00:11
       ---------------------- ----------------- 63.4/113.8 MB 4.9 MB/s eta 0:00:11
       ---------------------- ----------------- 64.7/113.8 MB 4.9 MB/s eta 0:00:10
       ----------------------- ---------------- 66.1/113.8 MB 5.0 MB/s eta 0:00:10
       ----------------------- ---------------- 67.1/113.8 MB 5.0 MB/s eta 0:00:10
       ----------------------- ---------------- 67.9/113.8 MB 5.0 MB/s eta 0:00:10
       ------------------------ --------------- 68.7/113.8 MB 4.9 MB/s eta 0:00:10
       ------------------------ --------------- 69.2/113.8 MB 4.9 MB/s eta 0:00:10
       ------------------------ --------------- 70.0/113.8 MB 4.9 MB/s eta 0:00:09
       ------------------------ --------------- 70.8/113.8 MB 4.9 MB/s eta 0:00:09
       ------------------------- -------------- 71.6/113.8 MB 4.8 MB/s eta 0:00:09
       ------------------------- -------------- 72.1/113.8 MB 4.8 MB/s eta 0:00:09
       ------------------------- -------------- 73.1/113.8 MB 4.8 MB/s eta 0:00:09
       ------------------------- -------------- 73.7/113.8 MB 4.8 MB/s eta 0:00:09
       -------------------------- ------------- 74.4/113.8 MB 4.8 MB/s eta 0:00:09
       -------------------------- ------------- 75.2/113.8 MB 4.8 MB/s eta 0:00:09
       -------------------------- ------------- 76.0/113.8 MB 4.8 MB/s eta 0:00:08
       --------------------------- ------------ 76.8/113.8 MB 4.7 MB/s eta 0:00:08
       --------------------------- ------------ 77.6/113.8 MB 4.7 MB/s eta 0:00:08
       --------------------------- ------------ 78.1/113.8 MB 4.7 MB/s eta 0:00:08
       --------------------------- ------------ 78.6/113.8 MB 4.7 MB/s eta 0:00:08
       --------------------------- ------------ 79.4/113.8 MB 4.7 MB/s eta 0:00:08
       ---------------------------- ----------- 80.2/113.8 MB 4.7 MB/s eta 0:00:08
       ---------------------------- ----------- 81.3/113.8 MB 4.7 MB/s eta 0:00:07
       ---------------------------- ----------- 82.1/113.8 MB 4.7 MB/s eta 0:00:07
       ----------------------------- ---------- 83.1/113.8 MB 4.6 MB/s eta 0:00:07
       ----------------------------- ---------- 84.1/113.8 MB 4.7 MB/s eta 0:00:07
       ----------------------------- ---------- 84.9/113.8 MB 4.7 MB/s eta 0:00:07
       ------------------------------ --------- 85.5/113.8 MB 4.6 MB/s eta 0:00:07
       ------------------------------ --------- 86.5/113.8 MB 4.6 MB/s eta 0:00:06
       ------------------------------ --------- 87.6/113.8 MB 4.6 MB/s eta 0:00:06
       ------------------------------- -------- 88.6/113.8 MB 4.6 MB/s eta 0:00:06
       ------------------------------- -------- 89.7/113.8 MB 4.6 MB/s eta 0:00:06
       ------------------------------- -------- 91.0/113.8 MB 4.6 MB/s eta 0:00:05
       -------------------------------- ------- 92.0/113.8 MB 4.7 MB/s eta 0:00:05
       -------------------------------- ------- 93.1/113.8 MB 4.7 MB/s eta 0:00:05
       --------------------------------- ------ 94.1/113.8 MB 4.7 MB/s eta 0:00:05
       --------------------------------- ------ 95.4/113.8 MB 4.7 MB/s eta 0:00:04
       --------------------------------- ------ 96.5/113.8 MB 4.7 MB/s eta 0:00:04
       ---------------------------------- ----- 97.8/113.8 MB 4.7 MB/s eta 0:00:04
       ---------------------------------- ----- 98.8/113.8 MB 4.7 MB/s eta 0:00:04
       ----------------------------------- ---- 100.1/113.8 MB 4.7 MB/s eta 0:00:03
       ----------------------------------- ---- 101.2/113.8 MB 4.7 MB/s eta 0:00:03
       ------------------------------------ --- 102.5/113.8 MB 4.7 MB/s eta 0:00:03
       ------------------------------------ --- 103.8/113.8 MB 4.7 MB/s eta 0:00:03
       ------------------------------------ --- 105.1/113.8 MB 4.8 MB/s eta 0:00:02
       ------------------------------------- -- 106.4/113.8 MB 4.8 MB/s eta 0:00:02
       ------------------------------------- -- 107.7/113.8 MB 4.8 MB/s eta 0:00:02
       -------------------------------------- - 109.1/113.8 MB 4.8 MB/s eta 0:00:01
       -------------------------------------- - 110.4/113.8 MB 4.8 MB/s eta 0:00:01
       ---------------------------------------  111.7/113.8 MB 4.8 MB/s eta 0:00:01
       ---------------------------------------  112.7/113.8 MB 4.8 MB/s eta 0:00:01
       ---------------------------------------  113.5/113.8 MB 4.8 MB/s eta 0:00:01
       ---------------------------------------  113.5/113.8 MB 4.8 MB/s eta 0:00:01
       ---------------------------------------- 113.8/113.8 MB 4.8 MB/s  0:00:23
    Using cached networkx-3.6.1-py3-none-any.whl (2.1 MB)
    Using cached sympy-1.14.0-py3-none-any.whl (6.3 MB)
    Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
    Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
    Using cached markupsafe-3.0.3-cp313-cp313-win_amd64.whl (15 kB)
    Using cached scikit_learn-1.8.0-cp313-cp313-win_amd64.whl (8.0 MB)
    Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
    Downloading scipy-1.17.1-cp313-cp313-win_amd64.whl (36.5 MB)
       ---------------------------------------- 0.0/36.5 MB ? eta -:--:--
        --------------------------------------- 0.5/36.5 MB 3.1 MB/s eta 0:00:12
       - -------------------------------------- 1.3/36.5 MB 3.1 MB/s eta 0:00:12
       -- ------------------------------------- 2.1/36.5 MB 3.4 MB/s eta 0:00:11
       -- ------------------------------------- 2.6/36.5 MB 3.4 MB/s eta 0:00:10
       --- ------------------------------------ 3.4/36.5 MB 3.3 MB/s eta 0:00:11
       ---- ----------------------------------- 4.2/36.5 MB 3.3 MB/s eta 0:00:10
       ----- ---------------------------------- 5.0/36.5 MB 3.4 MB/s eta 0:00:10
       ------ --------------------------------- 5.8/36.5 MB 3.5 MB/s eta 0:00:09
       ------ --------------------------------- 6.3/36.5 MB 3.4 MB/s eta 0:00:09
       ------- -------------------------------- 7.1/36.5 MB 3.4 MB/s eta 0:00:09
       -------- ------------------------------- 7.9/36.5 MB 3.4 MB/s eta 0:00:09
       --------- ------------------------------ 8.7/36.5 MB 3.5 MB/s eta 0:00:09
       ---------- ----------------------------- 9.4/36.5 MB 3.5 MB/s eta 0:00:08
       ----------- ---------------------------- 10.5/36.5 MB 3.6 MB/s eta 0:00:08
       ------------ --------------------------- 11.3/36.5 MB 3.6 MB/s eta 0:00:07
       ------------- -------------------------- 12.1/36.5 MB 3.7 MB/s eta 0:00:07
       -------------- ------------------------- 13.1/36.5 MB 3.7 MB/s eta 0:00:07
       --------------- ------------------------ 14.2/36.5 MB 3.8 MB/s eta 0:00:06
       ---------------- ----------------------- 14.9/36.5 MB 3.8 MB/s eta 0:00:06
       ----------------- ---------------------- 16.0/36.5 MB 3.9 MB/s eta 0:00:06
       ------------------ --------------------- 16.5/36.5 MB 3.8 MB/s eta 0:00:06
       ------------------- -------------------- 17.6/36.5 MB 3.9 MB/s eta 0:00:05
       -------------------- ------------------- 18.9/36.5 MB 3.9 MB/s eta 0:00:05
       --------------------- ------------------ 19.7/36.5 MB 4.0 MB/s eta 0:00:05
       ---------------------- ----------------- 20.7/36.5 MB 4.0 MB/s eta 0:00:04
       ----------------------- ---------------- 21.8/36.5 MB 4.0 MB/s eta 0:00:04
       ------------------------ --------------- 22.8/36.5 MB 4.1 MB/s eta 0:00:04
       -------------------------- ------------- 23.9/36.5 MB 4.1 MB/s eta 0:00:04
       --------------------------- ------------ 25.2/36.5 MB 4.2 MB/s eta 0:00:03
       ---------------------------- ----------- 26.2/36.5 MB 4.2 MB/s eta 0:00:03
       ----------------------------- ---------- 27.3/36.5 MB 4.2 MB/s eta 0:00:03
       ------------------------------- -------- 28.6/36.5 MB 4.3 MB/s eta 0:00:02
       -------------------------------- ------- 29.9/36.5 MB 4.4 MB/s eta 0:00:02
       --------------------------------- ------ 30.9/36.5 MB 4.4 MB/s eta 0:00:02
       ----------------------------------- ---- 32.0/36.5 MB 4.4 MB/s eta 0:00:02
       ------------------------------------ --- 33.3/36.5 MB 4.5 MB/s eta 0:00:01
       ------------------------------------- -- 34.6/36.5 MB 4.5 MB/s eta 0:00:01
       ---------------------------------------  35.9/36.5 MB 4.6 MB/s eta 0:00:01
       ---------------------------------------- 36.5/36.5 MB 4.5 MB/s  0:00:08
    Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
    Downloading typer-0.24.1-py3-none-any.whl (56 kB)
    Downloading annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
    Using cached click-8.3.1-py3-none-any.whl (108 kB)
    Downloading rich-14.3.3-py3-none-any.whl (310 kB)
    Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
    Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Downloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
    Installing collected packages: mpmath, threadpoolctl, sympy, shellingham, scipy, safetensors, numexpr, networkx, mdurl, MarkupSafe, joblib, hf-xet, fsspec, filelock, faiss-cpu, click, annotated-doc, scikit-learn, markdown-it-py, jinja2, torch, rich, typer, huggingface-hub, tokenizers, transformers, sentence-transformers

       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       ----------------------------------------  0/27 [mpmath]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       -- -------------------------------------  2/27 [sympy]
       ---- -----------------------------------  3/27 [shellingham]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ----- ----------------------------------  4/27 [scipy]
       ------- --------------------------------  5/27 [safetensors]
       -------- -------------------------------  6/27 [numexpr]
       -------- -------------------------------  6/27 [numexpr]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ---------- -----------------------------  7/27 [networkx]
       ----------- ----------------------------  8/27 [mdurl]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       -------------- ------------------------- 10/27 [joblib]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ----------------- ---------------------- 12/27 [fsspec]
       ------------------- -------------------- 13/27 [filelock]
       ------------------- -------------------- 13/27 [filelock]
       -------------------- ------------------- 14/27 [faiss-cpu]
       -------------------- ------------------- 14/27 [faiss-cpu]
       -------------------- ------------------- 14/27 [faiss-cpu]
       -------------------- ------------------- 14/27 [faiss-cpu]
       -------------------- ------------------- 14/27 [faiss-cpu]
       -------------------- ------------------- 14/27 [faiss-cpu]
       -------------------- ------------------- 14/27 [faiss-cpu]
       ---------------------- ----------------- 15/27 [click]
       ---------------------- ----------------- 15/27 [click]
       ---------------------- ----------------- 15/27 [click]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       ------------------------- -------------- 17/27 [scikit-learn]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       -------------------------- ------------- 18/27 [markdown-it-py]
       ---------------------------- ----------- 19/27 [jinja2]
       ---------------------------- ----------- 19/27 [jinja2]
       ---------------------------- ----------- 19/27 [jinja2]
       ---------------------------- ----------- 19/27 [jinja2]
       ---------------------------- ----------- 19/27 [jinja2]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ----------------------------- ---------- 20/27 [torch]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       ------------------------------- -------- 21/27 [rich]
       -------------------------------- ------- 22/27 [typer]
       -------------------------------- ------- 22/27 [typer]
       -------------------------------- ------- 22/27 [typer]
       -------------------------------- ------- 22/27 [typer]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ---------------------------------- ----- 23/27 [huggingface-hub]
       ----------------------------------- ---- 24/27 [tokenizers]
       ----------------------------------- ---- 24/27 [tokenizers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       ------------------------------------- -- 25/27 [transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       -------------------------------------- - 26/27 [sentence-transformers]
       ---------------------------------------- 27/27 [sentence-transformers]

    Successfully installed MarkupSafe-3.0.3 annotated-doc-0.0.4 click-8.3.1 faiss-cpu-1.13.2 filelock-3.25.2 fsspec-2026.2.0 hf-xet-1.4.2 huggingface-hub-1.7.1 jinja2-3.1.6 joblib-1.5.3 markdown-it-py-4.0.0 mdurl-0.1.2 mpmath-1.3.0 networkx-3.6.1 numexpr-2.14.1 rich-14.3.3 safetensors-0.7.0 scikit-learn-1.8.0 scipy-1.17.1 sentence-transformers-5.3.0 shellingham-1.5.4 sympy-1.14.0 threadpoolctl-3.6.0 tokenizers-0.22.2 torch-2.10.0 transformers-5.3.0 typer-0.24.1
    Note: you may need to restart the kernel to use updated packages.
:::
:::

::: {#259c74de .cell .code execution_count="7" vscode="{\"languageId\":\"shellscript\"}"}
``` python
pip install langchain==0.3.25 langchain-core==0.3.65 langchain-community==0.3.24 langchain-openai==0.3.16
```

::: {.output .stream .stdout}
    Collecting langchain==0.3.25
      Downloading langchain-0.3.25-py3-none-any.whl.metadata (7.8 kB)
    Collecting langchain-core==0.3.65
      Downloading langchain_core-0.3.65-py3-none-any.whl.metadata (5.8 kB)
    Collecting langchain-community==0.3.24
      Downloading langchain_community-0.3.24-py3-none-any.whl.metadata (2.5 kB)
    Collecting langchain-openai==0.3.16
      Downloading langchain_openai-0.3.16-py3-none-any.whl.metadata (2.3 kB)
    Collecting langchain-text-splitters<1.0.0,>=0.3.8 (from langchain==0.3.25)
      Downloading langchain_text_splitters-0.3.11-py3-none-any.whl.metadata (1.8 kB)
    Collecting langsmith<0.4,>=0.1.17 (from langchain==0.3.25)
      Downloading langsmith-0.3.45-py3-none-any.whl.metadata (15 kB)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain==0.3.25) (2.12.5)
    Collecting SQLAlchemy<3,>=1.4 (from langchain==0.3.25)
      Downloading sqlalchemy-2.0.48-cp313-cp313-win_amd64.whl.metadata (9.8 kB)
    Requirement already satisfied: requests<3,>=2 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain==0.3.25) (2.32.5)
    Requirement already satisfied: PyYAML>=5.3 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain==0.3.25) (6.0.3)
    Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain-core==0.3.65) (9.1.4)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain-core==0.3.65) (1.33)
    Collecting packaging<25,>=23.2 (from langchain-core==0.3.65)
      Downloading packaging-24.2-py3-none-any.whl.metadata (3.2 kB)
    Requirement already satisfied: typing-extensions>=4.7 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain-core==0.3.65) (4.15.0)
    Collecting aiohttp<4.0.0,>=3.8.3 (from langchain-community==0.3.24)
      Downloading aiohttp-3.13.3-cp313-cp313-win_amd64.whl.metadata (8.4 kB)
    Collecting dataclasses-json<0.7,>=0.5.7 (from langchain-community==0.3.24)
      Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
    Collecting pydantic-settings<3.0.0,>=2.4.0 (from langchain-community==0.3.24)
      Downloading pydantic_settings-2.13.1-py3-none-any.whl.metadata (3.4 kB)
    Collecting httpx-sse<1.0.0,>=0.4.0 (from langchain-community==0.3.24)
      Downloading httpx_sse-0.4.3-py3-none-any.whl.metadata (9.7 kB)
    Collecting numpy>=2.1.0 (from langchain-community==0.3.24)
      Downloading numpy-2.4.3-cp313-cp313-win_amd64.whl.metadata (6.6 kB)
    Collecting openai<2.0.0,>=1.68.2 (from langchain-openai==0.3.16)
      Downloading openai-1.109.1-py3-none-any.whl.metadata (29 kB)
    Requirement already satisfied: tiktoken<1,>=0.7 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langchain-openai==0.3.16) (0.12.0)
    Collecting aiohappyeyeballs>=2.5.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
    Collecting aiosignal>=1.4.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
    Collecting attrs>=17.3.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading attrs-25.4.0-py3-none-any.whl.metadata (10 kB)
    Collecting frozenlist>=1.1.1 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading frozenlist-1.8.0-cp313-cp313-win_amd64.whl.metadata (21 kB)
    Collecting multidict<7.0,>=4.5 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading multidict-6.7.1-cp313-cp313-win_amd64.whl.metadata (5.5 kB)
    Collecting propcache>=0.2.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading propcache-0.4.1-cp313-cp313-win_amd64.whl.metadata (14 kB)
    Collecting yarl<2.0,>=1.17.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community==0.3.24)
      Downloading yarl-1.23.0-cp313-cp313-win_amd64.whl.metadata (82 kB)
    Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community==0.3.24)
      Downloading marshmallow-3.26.2-py3-none-any.whl.metadata (7.3 kB)
    Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community==0.3.24)
      Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)
    Requirement already satisfied: jsonpointer>=1.9 in d:\miniconda3\envs\WYCity\Lib\site-packages (from jsonpatch<2.0,>=1.33->langchain-core==0.3.65) (3.0.0)
    INFO: pip is looking at multiple versions of langchain-text-splitters to determine which version is compatible with other requirements. This could take a while.
    Collecting langchain-text-splitters<1.0.0,>=0.3.8 (from langchain==0.3.25)
      Downloading langchain_text_splitters-0.3.10-py3-none-any.whl.metadata (1.9 kB)
      Downloading langchain_text_splitters-0.3.9-py3-none-any.whl.metadata (1.9 kB)
      Downloading langchain_text_splitters-0.3.8-py3-none-any.whl.metadata (1.9 kB)
    Requirement already satisfied: httpx<1,>=0.23.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langsmith<0.4,>=0.1.17->langchain==0.3.25) (0.28.1)
    Requirement already satisfied: orjson<4.0.0,>=3.9.14 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langsmith<0.4,>=0.1.17->langchain==0.3.25) (3.11.7)
    Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from langsmith<0.4,>=0.1.17->langchain==0.3.25) (1.0.0)
    Collecting zstandard<0.24.0,>=0.23.0 (from langsmith<0.4,>=0.1.17->langchain==0.3.25)
      Downloading zstandard-0.23.0-cp313-cp313-win_amd64.whl.metadata (3.0 kB)
    Requirement already satisfied: anyio in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.17->langchain==0.3.25) (4.12.1)
    Requirement already satisfied: certifi in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.17->langchain==0.3.25) (2026.2.25)
    Requirement already satisfied: httpcore==1.* in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.17->langchain==0.3.25) (1.0.9)
    Requirement already satisfied: idna in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.17->langchain==0.3.25) (3.11)
    Requirement already satisfied: h11>=0.16 in d:\miniconda3\envs\WYCity\Lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<0.4,>=0.1.17->langchain==0.3.25) (0.16.0)
    Requirement already satisfied: distro<2,>=1.7.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from openai<2.0.0,>=1.68.2->langchain-openai==0.3.16) (1.9.0)
    Requirement already satisfied: jiter<1,>=0.4.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from openai<2.0.0,>=1.68.2->langchain-openai==0.3.16) (0.13.0)
    Requirement already satisfied: sniffio in d:\miniconda3\envs\WYCity\Lib\site-packages (from openai<2.0.0,>=1.68.2->langchain-openai==0.3.16) (1.3.1)
    Requirement already satisfied: tqdm>4 in d:\miniconda3\envs\WYCity\Lib\site-packages (from openai<2.0.0,>=1.68.2->langchain-openai==0.3.16) (4.67.3)
    Requirement already satisfied: annotated-types>=0.6.0 in d:\miniconda3\envs\WYCity\Lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain==0.3.25) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.5 in d:\miniconda3\envs\WYCity\Lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain==0.3.25) (2.41.5)
    Requirement already satisfied: typing-inspection>=0.4.2 in d:\miniconda3\envs\WYCity\Lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain==0.3.25) (0.4.2)
    Collecting python-dotenv>=0.21.0 (from pydantic-settings<3.0.0,>=2.4.0->langchain-community==0.3.24)
      Downloading python_dotenv-1.2.2-py3-none-any.whl.metadata (27 kB)
    Requirement already satisfied: charset_normalizer<4,>=2 in d:\miniconda3\envs\WYCity\Lib\site-packages (from requests<3,>=2->langchain==0.3.25) (3.4.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in d:\miniconda3\envs\WYCity\Lib\site-packages (from requests<3,>=2->langchain==0.3.25) (2.6.3)
    Collecting greenlet>=1 (from SQLAlchemy<3,>=1.4->langchain==0.3.25)
      Downloading greenlet-3.3.2-cp313-cp313-win_amd64.whl.metadata (3.8 kB)
    Requirement already satisfied: regex>=2022.1.18 in d:\miniconda3\envs\WYCity\Lib\site-packages (from tiktoken<1,>=0.7->langchain-openai==0.3.16) (2026.2.28)
    Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain-community==0.3.24)
      Downloading mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)
    Requirement already satisfied: colorama in d:\miniconda3\envs\WYCity\Lib\site-packages (from tqdm>4->openai<2.0.0,>=1.68.2->langchain-openai==0.3.16) (0.4.6)
    Downloading langchain-0.3.25-py3-none-any.whl (1.0 MB)
       ---------------------------------------- 0.0/1.0 MB ? eta -:--:--
       ---------------------------------------- 1.0/1.0 MB 7.9 MB/s  0:00:00
    Downloading langchain_core-0.3.65-py3-none-any.whl (438 kB)
    Downloading langchain_community-0.3.24-py3-none-any.whl (2.5 MB)
       ---------------------------------------- 0.0/2.5 MB ? eta -:--:--
       ------------------------ --------------- 1.6/2.5 MB 7.0 MB/s eta 0:00:01
       ---------------------------------------- 2.5/2.5 MB 6.9 MB/s  0:00:00
    Downloading langchain_openai-0.3.16-py3-none-any.whl (62 kB)
    Downloading aiohttp-3.13.3-cp313-cp313-win_amd64.whl (453 kB)
    Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
    Downloading httpx_sse-0.4.3-py3-none-any.whl (9.0 kB)
    Downloading langchain_text_splitters-0.3.8-py3-none-any.whl (32 kB)
    Downloading langsmith-0.3.45-py3-none-any.whl (363 kB)
    Downloading marshmallow-3.26.2-py3-none-any.whl (50 kB)
    Downloading multidict-6.7.1-cp313-cp313-win_amd64.whl (45 kB)
    Downloading openai-1.109.1-py3-none-any.whl (948 kB)
       ---------------------------------------- 0.0/948.6 kB ? eta -:--:--
       ---------------------------------------- 948.6/948.6 kB 8.4 MB/s  0:00:00
    Downloading packaging-24.2-py3-none-any.whl (65 kB)
    Downloading pydantic_settings-2.13.1-py3-none-any.whl (58 kB)
    Downloading sqlalchemy-2.0.48-cp313-cp313-win_amd64.whl (2.1 MB)
       ---------------------------------------- 0.0/2.1 MB ? eta -:--:--
       ---------------------------------------- 2.1/2.1 MB 10.9 MB/s  0:00:00
    Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
    Downloading yarl-1.23.0-cp313-cp313-win_amd64.whl (87 kB)
    Downloading zstandard-0.23.0-cp313-cp313-win_amd64.whl (495 kB)
    Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
    Downloading aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
    Downloading attrs-25.4.0-py3-none-any.whl (67 kB)
    Downloading frozenlist-1.8.0-cp313-cp313-win_amd64.whl (43 kB)
    Downloading greenlet-3.3.2-cp313-cp313-win_amd64.whl (230 kB)
    Downloading mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
    Downloading numpy-2.4.3-cp313-cp313-win_amd64.whl (12.3 MB)
       ---------------------------------------- 0.0/12.3 MB ? eta -:--:--
       -------- ------------------------------- 2.6/12.3 MB 12.1 MB/s eta 0:00:01
       ----------------- ---------------------- 5.2/12.3 MB 12.2 MB/s eta 0:00:01
       ------------------------ --------------- 7.6/12.3 MB 11.8 MB/s eta 0:00:01
       ------------------------------ --------- 9.4/12.3 MB 11.3 MB/s eta 0:00:01
       -------------------------------------- - 11.8/12.3 MB 11.3 MB/s eta 0:00:01
       ---------------------------------------- 12.3/12.3 MB 10.1 MB/s  0:00:01
    Downloading propcache-0.4.1-cp313-cp313-win_amd64.whl (40 kB)
    Downloading python_dotenv-1.2.2-py3-none-any.whl (22 kB)
    Installing collected packages: zstandard, python-dotenv, propcache, packaging, numpy, mypy-extensions, multidict, httpx-sse, greenlet, frozenlist, attrs, aiohappyeyeballs, yarl, typing-inspect, SQLAlchemy, marshmallow, aiosignal, pydantic-settings, openai, langsmith, dataclasses-json, aiohttp, langchain-core, langchain-text-splitters, langchain-openai, langchain, langchain-community

      Attempting uninstall: zstandard

        Found existing installation: zstandard 0.25.0

        Uninstalling zstandard-0.25.0:

          Successfully uninstalled zstandard-0.25.0

       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       ----------------------------------------  0/27 [zstandard]
       - --------------------------------------  1/27 [python-dotenv]
       - --------------------------------------  1/27 [python-dotenv]
       -- -------------------------------------  2/27 [propcache]
      Attempting uninstall: packaging
       -- -------------------------------------  2/27 [propcache]
        Found existing installation: packaging 25.0
       -- -------------------------------------  2/27 [propcache]
        Uninstalling packaging-25.0:
       -- -------------------------------------  2/27 [propcache]
          Successfully uninstalled packaging-25.0
       -- -------------------------------------  2/27 [propcache]
       ---- -----------------------------------  3/27 [packaging]
       ---- -----------------------------------  3/27 [packaging]
       ---- -----------------------------------  3/27 [packaging]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       ----- ----------------------------------  4/27 [numpy]
       -------- -------------------------------  6/27 [multidict]
       ---------- -----------------------------  7/27 [httpx-sse]
       ----------- ----------------------------  8/27 [greenlet]
       ----------- ----------------------------  8/27 [greenlet]
       ----------- ----------------------------  8/27 [greenlet]
       ------------- --------------------------  9/27 [frozenlist]
       -------------- ------------------------- 10/27 [attrs]
       -------------- ------------------------- 10/27 [attrs]
       -------------- ------------------------- 10/27 [attrs]
       ----------------- ---------------------- 12/27 [yarl]
       ----------------- ---------------------- 12/27 [yarl]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       -------------------- ------------------- 14/27 [SQLAlchemy]
       ---------------------- ----------------- 15/27 [marshmallow]
       ---------------------- ----------------- 15/27 [marshmallow]
       ------------------------- -------------- 17/27 [pydantic-settings]
       ------------------------- -------------- 17/27 [pydantic-settings]
       ------------------------- -------------- 17/27 [pydantic-settings]
       ------------------------- -------------- 17/27 [pydantic-settings]
      Attempting uninstall: openai
       ------------------------- -------------- 17/27 [pydantic-settings]
        Found existing installation: openai 2.28.0
       ------------------------- -------------- 17/27 [pydantic-settings]
       -------------------------- ------------- 18/27 [openai]
        Uninstalling openai-2.28.0:
       -------------------------- ------------- 18/27 [openai]
          Successfully uninstalled openai-2.28.0
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
       -------------------------- ------------- 18/27 [openai]
      Attempting uninstall: langsmith
       -------------------------- ------------- 18/27 [openai]
        Found existing installation: langsmith 0.7.17
       -------------------------- ------------- 18/27 [openai]
        Uninstalling langsmith-0.7.17:
       -------------------------- ------------- 18/27 [openai]
          Successfully uninstalled langsmith-0.7.17
       -------------------------- ------------- 18/27 [openai]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ---------------------------- ----------- 19/27 [langsmith]
       ----------------------------- ---------- 20/27 [dataclasses-json]
       ----------------------------- ---------- 20/27 [dataclasses-json]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
       ------------------------------- -------- 21/27 [aiohttp]
      Attempting uninstall: langchain-core
       ------------------------------- -------- 21/27 [aiohttp]
        Found existing installation: langchain-core 1.2.19
       ------------------------------- -------- 21/27 [aiohttp]
       -------------------------------- ------- 22/27 [langchain-core]
        Uninstalling langchain-core-1.2.19:
       -------------------------------- ------- 22/27 [langchain-core]
          Successfully uninstalled langchain-core-1.2.19
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       -------------------------------- ------- 22/27 [langchain-core]
       ---------------------------------- ----- 23/27 [langchain-text-splitters]
       ---------------------------------- ----- 23/27 [langchain-text-splitters]
      Attempting uninstall: langchain-openai
       ---------------------------------- ----- 23/27 [langchain-text-splitters]
        Found existing installation: langchain-openai 1.1.11
       ---------------------------------- ----- 23/27 [langchain-text-splitters]
        Uninstalling langchain-openai-1.1.11:
       ---------------------------------- ----- 23/27 [langchain-text-splitters]
          Successfully uninstalled langchain-openai-1.1.11
       ---------------------------------- ----- 23/27 [langchain-text-splitters]
       ----------------------------------- ---- 24/27 [langchain-openai]
       ----------------------------------- ---- 24/27 [langchain-openai]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       ------------------------------------- -- 25/27 [langchain]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       -------------------------------------- - 26/27 [langchain-community]
       ---------------------------------------- 27/27 [langchain-community]

    Successfully installed SQLAlchemy-2.0.48 aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-1.4.0 attrs-25.4.0 dataclasses-json-0.6.7 frozenlist-1.8.0 greenlet-3.3.2 httpx-sse-0.4.3 langchain-0.3.25 langchain-community-0.3.24 langchain-core-0.3.65 langchain-openai-0.3.16 langchain-text-splitters-0.3.8 langsmith-0.3.45 marshmallow-3.26.2 multidict-6.7.1 mypy-extensions-1.1.0 numpy-2.4.3 openai-1.109.1 packaging-24.2 propcache-0.4.1 pydantic-settings-2.13.1 python-dotenv-1.2.2 typing-inspect-0.9.0 yarl-1.23.0 zstandard-0.23.0
    Note: you may need to restart the kernel to use updated packages.
:::

::: {.output .stream .stderr}
      WARNING: Failed to remove contents in a temporary directory 'D:\miniconda3\envs\WYCity\Lib\site-packages\~standard'.
      You can safely remove it manually.
:::
:::

::: {#828aed65 .cell .markdown}
这之后要告诉编辑器有这个东西（以VScode为例）\
1.按 Ctrl + Shift + P。\
2.输入 Python: Select Interpreter。\
3.选择列表里带有 venv 字样的那个（或者选 Enter interpreter path -\>
找到你项目文件夹下的 venv/Scripts/python.exe）。

## 二、搭建智能体的可用工具列表：一个数理知识库和一个数学计算工具

数理知识库提供模型可以读取的信息，数学计算工具就顾名思义了，程序来算肯定比大模型自己来要快
:::

::: {#358d65f6 .cell .code execution_count="9"}
``` python
# 这是数理知识库的构建过程
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # 根据不同API更换
from langchain.tools import Tool

# 第1步: 直接加载原始 PDF
# 使用 PyPDFLoader 读取 PDF 文件。这里假设 PDF 文件名为 "数理公式 (全).pdf"
loader = PyPDFLoader("数理公式 (全).pdf")

# 使用 loader.load() 方法加载 PDF 文件中的所有页面，并将其存储在 raw_pages 变量中
raw_pages = loader.load()

# 第2步: 直接按字符数切分
# 创建一个文本分割器，名为 RecursiveCharacterTextSplitter，其目的是分割文档为更小的部分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个切分的片段最大为500个字符
    chunk_overlap=50 # 片段之间有50个字符的重叠以防止公式被切断
)

# 使用创建的文本分割器将加载的 PDF 文件内容分割成多个 smaller documents
docs = text_splitter.split_documents(raw_pages)

# 第3步: 建立搜索索引
# 创建一个 FAISS 向量存储。在计算文档向量时使用 OpenAIEmbeddings，以便进行快速的相似性搜索
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 第4步: 定义直接检索工具
# 创建一个搜索函数，用于执行快速搜索任务
def quick_search(query: str):
    # 使用 FAISS 向量存储执行相似性搜索，返回与查询最相似的3个文档片段
    search_results = vectorstore.similarity_search(query, k=3)
    # 将搜索结果汇总为一个字符串，结果片段之间用 "---" 分隔
    return "\n---\n".join([res.page_content for res in search_results])

# 定义一个工具来进行快速参考
reference_tool = Tool(
    name="Quick_Reference",
    func=quick_search,
    description="直接搜索 PDF 文档中的原始文本。当你需要寻找特定的物理公式或数学定义时使用。"
)

# 现在，可以使用 reference_tool 执行快速搜索任务，以查找特定的物理公式或数学定义。
```

::: {.output .error ename="ValueError" evalue="File path 数理公式 (全).pdf is not a valid file or url"}
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    Cell In[9], line 10
          6 from langchain.tools import Tool
          8 # 第1步: 直接加载原始 PDF
          9 # 使用 PyPDFLoader 读取 PDF 文件。这里假设 PDF 文件名为 "数理公式 (全).pdf"
    ---> 10 loader = PyPDFLoader("数理公式 (全).pdf")
         12 # 使用 loader.load() 方法加载 PDF 文件中的所有页面，并将其存储在 raw_pages 变量中
         13 raw_pages = loader.load()

    File d:\miniconda3\envs\WYCity\Lib\site-packages\langchain_community\document_loaders\pdf.py:281, in PyPDFLoader.__init__(self, file_path, password, headers, extract_images, mode, images_parser, images_inner_format, pages_delimiter, extraction_mode, extraction_kwargs)
        238 def __init__(
        239     self,
        240     file_path: Union[str, PurePath],
       (...)    250     extraction_kwargs: Optional[dict] = None,
        251 ) -> None:
        252     """Initialize with a file path.
        253 
        254     Args:
       (...)    279         `aload` methods to retrieve parsed documents with content and metadata.
        280     """
    --> 281     super().__init__(file_path, headers=headers)
        282     self.parser = PyPDFParser(
        283         password=password,
        284         mode=mode,
       (...)    290         extraction_kwargs=extraction_kwargs,
        291     )

    File d:\miniconda3\envs\WYCity\Lib\site-packages\langchain_community\document_loaders\pdf.py:140, in BasePDFLoader.__init__(self, file_path, headers)
        138         self.file_path = str(temp_pdf)
        139 elif not os.path.isfile(self.file_path):
    --> 140     raise ValueError("File path %s is not a valid file or url" % self.file_path)

    ValueError: File path 数理公式 (全).pdf is not a valid file or url
:::
:::

::: {#3f17ecdb .cell .code}
``` python
# 这是数学计算工具的搭建过程
import sys
from io import StringIO
from langchain.tools import tool

@tool
def physics_math_solver(query: str):
    """
    一个数理逻辑计算引擎。
    输入：一段符合 Python 语法的数学计算代码。
    功能：利用 sympy 处理符号运算（求导、积分、方程求解）或利用 math 处理数值计算。
    """
    # 第1步: 准备一个“沙盒”环境，防止直接操作主程序
    # 创建一个安全的全局环境字典，包含常用的数学库，避免用户需要重复导入
    safe_globals = {
        "sympy": __import__("sympy"),  # 导入 sympy 进行符号数学运算
        "math": __import__("math"),    # 导入 math 处理基本数学运算
        "np": __import__("numpy")      # 导入 numpy 进行更多的数值计算
    }
    
    # 第2步: 捕获代码运行的输出结果
    # 备份当前的标准输出位置，以便恢复
    old_stdout = sys.stdout
    # 重定向标准输出到一个字符串缓冲区
    redirected_output = sys.stdout = StringIO()
    
    try:
        # 使用 exec() 函数在指定的安全全局环境中执行用户提供的代码
        exec(query, safe_globals)
        # 代码执行完成后，恢复标准输出
        sys.stdout = old_stdout
        # 返回执行代码后的输出结果
        return redirected_output.getvalue()
    except Exception as e:
        # 若代码执行过程中发生异常，恢复标准输出
        sys.stdout = old_stdout
        # 返回错误信息
        return f"代码执行出错: {str(e)}"
```
:::

::: {#92368f48 .cell .markdown}
## 三、检索式问答链和ReAct范式智能体的创建

这个的基本概念已经讨论过了，下面给出可运行的代码\
共分为两段，第一段是我用的一个txt形式的知识库创建的代码，这也可以看出知识库构建形式有很多，第二段是整体可运行程序（记得先运行第一段哦）\
知识库可以从这里查看：<https://cloud.tsinghua.edu.cn/d/016f473aafe444d1aae5/>
什么？你问我为什么写这么长？还不是因为我让deepseek输出"final
answer"它老输出"最终答案"导致必须加特判嘛
:::

::: {#2f90f1c9 .cell .code}
``` python
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = "knowledge.txt"
db_save_path = "faiss_index"

print(f"正在读取文件: {file_path}...")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=40)
docs = text_splitter.split_documents(documents)

print(f"数据加载完成，共切分为 {len(docs)} 条知识片段。")
print("正在向量化并构建索引...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(db_save_path)

print(f"✅ 知识库已保存到: {db_save_path}")
```
:::

::: {#a76929ae .cell .code}
``` python
import os
import re
import sys
from io import StringIO

from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# =======================================================
# 第一步：配置大模型
# =======================================================
api_key_file = "api_key.txt"
if not os.path.exists(api_key_file):
    raise FileNotFoundError("未找到 api_key.txt，请在项目根目录创建该文件并写入你的 API Key。")

with open(api_key_file, "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()

if not API_KEY:
    raise ValueError("api_key.txt 为空，请写入有效的 API Key。")

llm = ChatOpenAI(
    model="DeepSeek-V3.2",
    api_key=API_KEY,
    base_url="https://llmapi.paratera.com",
    temperature=0,
    streaming=True
)

# =======================================================
# 第二步：加载知识库（FAISS + PDF）
# =======================================================

# ---------- FAISS 知识库 ----------
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
        return f"代码执行出错: {str(e)}"

# 组装工具列表（根据 PDF 是否加载动态增减）
tools = [
    Tool(
        name="Knowledge_Base",
        func=retrieve_knowledge,
        description=(
            "查询预建的物理定律、公式、概念知识库。"
            "输入是具体的中文问题，例如：牛顿第二定律是什么？"
        )
    ),
    Tool(
        name="Math_Solver",
        func=physics_math_solver,
        description=(
            "数理计算引擎，支持符号运算和数值计算。"
            "输入必须是合法的 Python 代码，使用 print() 输出结果。"
            "可直接使用 sympy、math、np、symbols、solve、diff、integrate 等。"
            "示例：print(solve('x**2 - 4', symbols('x'))) 或 print(math.sqrt(9))"
        )
    ),
]

if pdf_vectorstore is not None:
    tools.insert(1, Tool(
        name="PDF_Reference",
        func=pdf_quick_search,
        description=(
            "直接搜索《数理公式》PDF 文档中的原始文本。"
            "当需要查找特定物理公式或数学定义的原文时使用。"
            "输入是关键词或问题描述。"
        )
    ))

# =======================================================
# 第四步：容错解析器
# =======================================================
FINAL_ANSWER_TRIGGERS = [
    "Final Answer:",
    "最终答案",
    "**最终答案",
    "final answer:",
    "最终回答",
    "答案：",
    "答案:",
]

class ForgivingOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        for trigger in FINAL_ANSWER_TRIGGERS:
            if trigger in text:
                answer = text.split(trigger, 1)[-1].strip().strip("*").strip()
                return AgentFinish(return_values={"output": answer}, log=text)
        # 兜底：内容完整但无 Action，直接返回
        if len(text.strip()) > 50 and "Action:" not in text:
            return AgentFinish(return_values={"output": text.strip()}, log=text)
        return super().parse(text)

# =======================================================
# 第五步：构建 ReAct Agent
# =======================================================
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT RULES:
- For Math_Solver, Action Input must be valid Python code that uses print() to output results.
- Always end your response with "Final Answer:" followed by your complete answer.
- Do NOT use markdown headers or **bold** before "Final Answer:".
- If you already know the answer, go directly to "Final Answer:".

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(prompt_template)
agent = create_react_agent(llm, tools, prompt, output_parser=ForgivingOutputParser())

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="请严格按照格式，以 'Final Answer:' 结尾。",
    max_iterations=10
)

# =======================================================
# 第六步：交互式运行
# =======================================================
if __name__ == "__main__":
    tool_names = [t.name for t in tools]
    print("\n" + "="*50)
    print(f">>> 数理全能王智能体已就绪！可用工具: {tool_names}")
    print(">>> 示例问题：")
    print("    1. 根据牛顿第二定律，质量5kg加速度2m/s^2，力是多少？")
    print("    2. 求 x^2 - 5x + 6 = 0 的解")
    print("    3. 对 sin(x) 求导")
    print("    (输入 'exit' 退出)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("用户提问: ").strip()
            if user_input.lower() in ["exit", "quit", "q", "退出"]:
                print("再见！")
                break
            if not user_input:
                continue
            print("-" * 30)
            result = agent_executor.invoke({"input": user_input})
            print("-" * 30)
            print(f"最终答案: {result['output']}")
            print("=" * 50 + "\n")
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请尝试换个问法。\n")
```
:::
