from dotenv import load_dotenv
load_dotenv()

import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = ChatNVIDIA(
    model="meta/llama-4-maverick-17b-128e-instruct",
    temperature=0.7
)

chat_history = []

query = input("User: ")

chat_history.append(HumanMessage(content=query))

messages = [
    SystemMessage(content="""
You are a research planner.

Break the user's question into 5 smaller research questions
that will help gather complete information.

Return them as a numbered list.
""")
] + chat_history

sub_questions_text = model.invoke(messages).content

chat_history.append(AIMessage(content=sub_questions_text))

print("\nGenerated Sub Questions:\n", sub_questions_text)

search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_KEY"))

sub_questions = [
    q.strip() for q in sub_questions_text.split("\n") if q.strip()
]

links = []

for q in sub_questions:
    try:
        results = search.results(q)

        for r in results["organic_results"][:2]:
            links.append(r["link"])

    except Exception as e:
        print("Search error:", e)

# remove duplicate links
links = list(set(links))

print("\nCollected Links:")
for l in links:
    print(l)

loader = WebBaseLoader(links[:5])
docs = loader.load()

data = []

for doc in docs:
    text = doc.page_content.replace("\n", " ")
    data.append(text[:1000])

context = "\n\n".join(data)

messages = [
    SystemMessage(content="""
You are a research assistant.

Using the provided webpage data, create a structured report.

Return output in this format:

Topic
Key Insights
Important Trends
Sources
Conclusion
"""),

    HumanMessage(content=f"""
User Query:
{query}

Data:
{context}

Sources:
{links}
""")
]

response = model.invoke(messages)

chat_history.append(AIMessage(content=response.content))

print("\nFinal Research Report:\n")
print(response.content)