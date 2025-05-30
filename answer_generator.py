import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(question, context_chunks, image_b64=None):
    context = "\n\n".join(context_chunks)
    image_note = ""
    if image_b64:
        image_note = "\n\n(Note: A screenshot related to the question is also provided.)"

    prompt = f"""You are a helpful teaching assistant. Use the context below to answer the question precisely.
1. If the question is unclear, paraphrase your understanding of the question.
2. Cite all relevant sections from `tds-content.xml` or `ga*.md`. Begin with: "According to [this reference](https://tds.s-anand.net/#/...), ...". Cite ONLY from the relevant <source>. ALWAYS cite verbatim. Mention ALL material relevant to the question.
3. Search online for additional answers. Share results WITH CITATION LINKS.
4. Think step-by-step. Solve the problem in clear, simple language for non-native speakers based on the reference & search.
5. Follow-up: Ask thoughtful questions to help students explore and learn.

Context:
{context}
{image_note}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

