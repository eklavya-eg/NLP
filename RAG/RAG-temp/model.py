from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from helper import mkLogger

MODEL_PATH = "D:\Models\TheBloke\llama-2-7b-chat.ggmlv3.q8_0.bin"

DB_FAISS_PATH = "vectorstores/db_faiss"
DB_FAISS_PATH1 = "vectorstores/db_faiss1"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])
    return prompt
    

def load_llm():
    # """Loads the Llama-2 LLM model."""
    model_name = "llama-2-7b-chat.ggmlv3.q8_0.bin" #llama-2-7b-chat.ggmlv3.q8_0.bin
    model_name = MODEL_PATH #llama-2-7b-chat.ggmlv3.q8_0.bin
    model_type = "llama"
    max_new_tokens = 30
    temperature = 0.2
    llm = CTransformers(model=model_name, model_type=model_type, max_new_tokens=max_new_tokens, temperature=temperature)
    return llm

def retrieval_qa_chain(llm, prompt, db):
    # """Creates a retrieval-QA chain for question answering."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs = {'prompt': prompt})
    return qa_chain

def qa_bot():
    # """Sets up the question-answering bot."""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    db1 = FAISS.load_local(DB_FAISS_PATH1, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    qa1 = retrieval_qa_chain(llm, qa_prompt, db1)
    return (qa, qa1)

def result(query):
    qa_result = qa_bot()
    qa, qa1 = qa_result  # Extract both QA systems from the tuple
    
    response_qa = qa({'query': query})
    response_qa1 = qa1({'query': query})
    
    return (response_qa, response_qa1)

if __name__=="__main__":
    logger = mkLogger("query_result.log")
    query = "distribution channels and percentage in apple"
    res = result(query)
    print(res)
    res1 = res[1]
    res = res[0]
    logger.info(f"query: {res['query']}")
    logger.info(f"result: {res['result']}")
    logger.info(f"result: {res1['result']}")
