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

# Source for Llama usage:
# from https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/model.py

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. In the following chat, you are first confronted with a piece of information that's been retrieved from a document.
The user will then ask questions about the piece of information. If the question is not related to the provided information, please communicate this issue to the user. \
"""

chat = []

# Use this for Llama (modifiedm, based on: https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/model.py)
def get_prompt(retrieved_source: str, message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    initial_text = f'<s>[INST] <<SYS>>\n{system_prompt}\nPiece of information: {retrieved_source}\n<</SYS>>\n\n'
    texts = [initial_text]
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    prompt = ''.join(texts) 
    print(f">>>>Prompt: {prompt}")
    return prompt 


def extract_text_from_storage_result(res_list):
    text = ""

    for d in res_list:
        content = d.page_content
        content = content.replace("\n", "")
        text += content 

    return text
    
    
if __name__ == '__main__':
        pdf = open("bonus_system_EN.pdf", "rb")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
          text += page.extract_text()
          
        # split into chunks
        text_splitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=500,
          chunk_overlap=0,
          length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        # embeddings = OpenAIEmbeddings()

        embeddings = GPT4AllEmbeddings()
        # TODO extend this for gpu usage!
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        model_name = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True)
        llm = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', trust_remote_code=True)

        model = pipeline("text-generation", model=llm, tokenizer=tokenizer, max_length=2048)

        user_question = input("Ask a question about your PDF: ")
        while user_question != "<<STOP>>":

            docs = knowledge_base.similarity_search(user_question, k=3)
            docs = extract_text_from_storage_result(docs)

            inputs = get_prompt(retrieved_source=docs, message=user_question, chat_history=[("", "")], system_prompt=DEFAULT_SYSTEM_PROMPT)
            output = model(inputs, return_full_text=False)
            response = output[0]["generated_text"]
            print(f"# Llama2: {response}")
            chat.append((user_question, response))
            # TODO create chat history
            follow_up = input("Your answer: ")
            while follow_up != "<<NEW>>":
                docs = knowledge_base.similarity_search(user_question, k=8)
                docs = extract_text_from_storage_result(docs)
                inputs = get_prompt(retrieved_source=docs, message=follow_up, chat_history=chat, system_prompt=DEFAULT_SYSTEM_PROMPT)
                output = model(inputs, return_full_text=False)
                response = output[0]["generated_text"]
                print(f"# Llama2: {response}")
                chat.append((follow_up, response))
                follow_up = input("Your answer: ")

            user_question = input("Ask a question about your PDF: ")
