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
from langchain.chains import RetrievalQA
from langchain.retrievers.arxiv import ArxivRetriever
import langchain


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

        model_name = "OpenAssistant/falcon-7b-sft-mix-2000"
        tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', trust_remote_code=True)

        pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000) 

        llm = HuggingFacePipeline(pipeline=pipeline)

        llm_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=knowledge_base.as_retriever()) 
        # llm_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ArxivRetriever()) 
        

        # llm = HuggingFacePipeline.from_model_id(
        #     model_id="OpenAssistant/falcon-7b-sft-mix-2000",
        #     task="text-generation",
        #     model_kwargs={"trust_remote_code":True, "max_length":2000}
        #   )

        # llm = HuggingFacePipeline.from_model_id(
          #     model_id="EleutherAI/pythia-1b-deduped",
          #     task="text-generation",
          #     device=0,
          #     model_kwargs={"trust_remote_code": True, "max_length": 1500}
          # )

        # show user input
        user_question = input("Ask a question about your PDF: ")
        while user_question != "STOP":

          response = llm_chain.run(user_question)

          print(">>> ANSWER: <<<")
          print(response["text"]) 
          print(30*"-")

          user_question = input("Ask a question about your PDF: ")
