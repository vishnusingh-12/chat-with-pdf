# !!! For GPU llama cpp needs some additional steps for installation. Check the below link for reference.
# https://python.langchain.com/docs/integrations/llms/llamacpp

# importing required libraries
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain


# function to load locally available llama 2 model. The link to download is:
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf
# you can download the 13b model as well according to your hardware

def load_llm():
    print("loading llm")
    """
    Loads the llm using LlamaCpp and returns the llm.
    """
    llm = LlamaCpp(
        # path to your local llama2 model
        model_path='D:\ML Resources\Projects\Langchain\chatbot\llama-2-7b-chat.Q8_0.gguf',

        # maximum number of tokens (input + output) that the model can generate
        max_tokens=1024,

        # number of gpu layers to be used, -1 for all layers, avoid in case of cpu loading
        n_gpu_layers=-1,

        # allows the model to offload the computation of key, query, and value vectors for certain layers
        offload_kqv=True,

        # the number of tokens the model can take as input at a time
        n_batch=256,

        # maximum tokens(input + output) that the model can process in one go. Max is 4096
        n_ctx=2048,

        # print the details of the execution of the llm
        verbose=True,

        # set for sampling tokens. High temp means less predictable output
        temperature=0.0
    )
    print('llm loaded')
    return llm


# loading llm before running chainlit app as it will be required throughout the program
llm_model = load_llm()


def convo_chain(db):
    """
    Creates and returns a Conversational Retrieval chain
    :param db: Vector database variable (Chroma db)
    :return:  Conversational Chain
    """
    # retriever uses the vector store(db) to retrieve documents that we will feed
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=db.as_retriever(),
                                                     return_source_documents=True)
    return qa_chain


# chainlit code

# decorator showing start() function will be called when chat starts
@cl.on_chat_start
async def start():
    # sends an opening message
    await cl.Message(content="Hello there, Welcome to chat with your data!").send()

    # declaring a none type variable for file upload
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!(Please be patient, this might take a few minutes.)",
            accept=["application/pdf"],
            max_size_mb=50
        ).send()

    # getting the uploaded file
    file = files[0]

    # creating loader object with file path
    loader = PyPDFLoader(file.path)

    # loading the contents of the pdf file
    documents = loader.load()

    # splitting the pdf text to small chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents=documents)

    # creating embeddings for the text using Hugging Face embeddings (384 dimensions)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # using chromadb for storing embeddings
    db = Chroma.from_documents(chunks, embeddings)

    # creating chain
    chain = convo_chain(db)

    # I am ready message to tell the user that processing has finished
    await cl.Message("I am ready").send()

    # setting session variable
    cl.user_session.set("chain", chain)

    # setting another variable history to store one previous question
    chat_history = []
    cl.user_session.set("history", chat_history)


# decorator showing that main() function will be called on message send
@cl.on_message
async def main(message: cl.Message):

    # getting the session variable chain
    chain = cl.user_session.get("chain")

    # creating a callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # getting result from chain
    res = await chain.acall({'question': message.content, 'chat_history': cl.user_session.get('history')},
                            callbacks=[cb])

    # getting answer from chain output
    answer = res["answer"]
    sources = res["source_documents"]

    if sources:
        pass
    else:
        answer += "\nNo sources found"

    # passing only previous question for chat history
    cl.user_session.set('history', [(message.content, '')])
