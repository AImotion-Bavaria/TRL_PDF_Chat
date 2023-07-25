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
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


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

        pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024) 

        llm = HuggingFacePipeline(pipeline=pipeline)

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
        while user_question != "<|STOP|>":

            docs = knowledge_base.similarity_search(user_question, k=8)
            docs = extract_text_from_storage_result(docs)
            print(f"Retrieved text: {docs}")

            initial_prompt = f"""<|prompter|> Consider the following pieces of information: {docs}.
                            Answer the following question about the pieces of information: {user_question}.
                            If the question is not related to the provided information, please communicate this issue to the user.
                            <|endoftext|><|assistant|>"""

            conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory(human_prefix="<|prompter|>", ai_prefix="<|assistant|>")) 

            print(conversation.predict(input=initial_prompt))

            follow_up = input("Your answer: ")
            while follow_up != "<|NEW|>":
                print(conversation.predict(input=follow_up))
                follow_up = input("Your answer: ")

            user_question = input("Ask a question about your PDF: ")
