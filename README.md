# Chat with pdfs

Talk to a chatbot about your pdf. Ask questions, get sumarization, insights etc. Chatbot also remebers the chat history.

# Features

- Can chat about about your pdf document. Document can be uploaded easily from a local machine. (Will add document reading from links soon)
- Utilizes RAG (Retrieval-Augmented Generation) with the provided pdf document.
- Uses HuggingFace embeddings for vector conversion.
- Uses Chromadb to create and store a vector index.
- Uses the llama 2 LLM model for generating natural response.

## Screenshots

![App Screenshots](https://raw.githubusercontent.com/vishnusingh-12/chat-with-pdf/main/readme/cbot2.PNG)
![App Screenshots](https://raw.githubusercontent.com/vishnusingh-12/chat-with-pdf/main/readme/cbot.PNG)




## Deployment
<pre>git clone  https://github.com/vishnusingh-12/chat-with-pdf
cd chat-with-pdf
pip install -r requirements.txt
chainlit run main.py </pre>

The model uses GPU with CUDA 11.8 to run for which LlamaCpp has to be configured differently. Follow the below link for reference:
https://python.langchain.com/docs/integrations/llms/llamacpp

The model uses a quantized llama 2 model and can also run on cpu with no change in the code. If LlamaCpp Cuda version is available it uses GPU else it uses CPU. 


## Technologies
<img src="https://raw.githubusercontent.com/vishnusingh-12/chat-with-pdf/main/readme/chatwithpdf.PNG">

## Support

For support, contact me:

[<img src="https://img.icons8.com/color/48/000000/gmail.png" alt="Gmail" width="30" height="30">](mailto:vishnusingh1995@gmail.com)
&nbsp;&nbsp;&nbsp;
[<img src="https://img.icons8.com/color/48/000000/instagram-new.png" alt="Instagram" width="30" height="30">](https://www.instagram.com/vishnusingh12/)
&nbsp;&nbsp;&nbsp;
[<img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub" width="30" height="30">](https://github.com/vishnusingh-12)
&nbsp;&nbsp;&nbsp;
[<img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn" width="30" height="30">](https://www.linkedin.com/in/singh-vishnu)

