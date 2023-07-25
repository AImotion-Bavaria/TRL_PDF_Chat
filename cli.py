from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate, LLMChain
import time


def extract_text_from_storage_result(res_list):
    text = ""

    for d in res_list:
        content = d.page_content
        content = content.replace("\n", "")
        text += content 

    return text
    
    
    # extract the text
    
if __name__ == '__main__':
        pdf = open("bonus_system_EN.pdf", "rb")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
          text += page.extract_text()
          
        # split into chunks
        text_splitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=250,
          chunk_overlap=0,
          length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        # embeddings = OpenAIEmbeddings()

        embeddings = GPT4AllEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # model_name = "OpenAssistant/falcon-7b-sft-mix-2000"

        model_name = "meta-llama/Llama-2-7b-chat-hf"

        # 13b is too large :(
        # model_name = "meta-llama/Llama-2-13b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', trust_remote_code=True, use_auth_token=True)

        pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024) 

        llm = HuggingFacePipeline(pipeline=pipeline)

        # show user input
        user_question = input("Ask a question about your PDF: ")
        while user_question != "STOP":

          docs = knowledge_base.similarity_search(user_question, k=8)

          docs = extract_text_from_storage_result(docs)

          template = """<|prompter|> Consider the following pieces of information: {context}.
          Answer the following question about the pieces of information: {user_question}.
          If the question is not related to the provided information, please communicate this issue to the user.<|endoftext|><|assistant|>"""

          prompt = PromptTemplate(template=template, input_variables=["context", "user_question"])
          llm_chain = LLMChain(prompt=prompt, llm=llm) 
          response = llm_chain({"context": docs, "user_question": user_question})

          print(">>> ANSWER: <<<")
          print(30*"-")
          print(f"Retrieved text: {docs}")
          print(30*"-")
          print(response["text"]) 
          print(30*"-")

          user_question = input("Ask a question about your PDF: ")
