# Add this at the very top
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import streamlit as st

# Load environment variables
load_dotenv()

# Get API key from environment or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

# Define instructions template
INSTRUCTIONS = """
You are a helpful assistant and you work as a legal advisor.
You should explain the topic in simple words so the user understands.
Always include the section and purpose of the law when possible.

Context:
{context}

Question: {question}

If the context does not contain the answer, say:
"Information not available in the provided context."
"""

class LegalAdvisorBot:
    def __init__(self, pdf_path="law_chatbot/law1.pdf"):
        self.pdf_path = pdf_path
        self.llm = None
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.memory = None
        self.qa_chain = None
        self.prompt = None
        self._initialize_components()

    def _initialize_components(self):
        try:
            # Initialize LLM
            if not GROQ_API_KEY:
                st.warning("GROQ_API_KEY is not set. Using dummy mode.")
                # Create a dummy LLM for initialization
                from langchain.schema import BaseLLM
                class DummyLLM(BaseLLM):
                    def _generate(self, prompts, stop=None):
                        return ["I'm a dummy LLM. Please set GROQ_API_KEY."] * len(prompts)
                    @property
                    def _llm_type(self):
                        return "dummy"
                self.llm = DummyLLM()
            else:
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    max_tokens=None,
                    api_key=GROQ_API_KEY
                )

            # Embeddings with error handling
            try:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e:
                st.error(f"Embedding model loading failed: {str(e)}")
                # Fallback to dummy embeddings
                from langchain.embeddings.base import Embeddings
                class DummyEmbeddings(Embeddings):
                    def embed_documents(self, texts):
                        return [[0.1] * 384] * len(texts)
                    def embed_query(self, text):
                        return [0.1] * 384
                self.embedding_model = DummyEmbeddings()

            # Load or create vector store
            vector_store_path = "faiss_index"
            try:
                if os.path.exists(vector_store_path):
                    self.vector_store = FAISS.load_local(
                        vector_store_path, 
                        self.embedding_model, 
                        allow_dangerous_deserialization=True
                    )
                    st.success("Loaded existing vector store.")
                else:
                    if os.path.exists(self.pdf_path):
                        loader = PyPDFLoader(self.pdf_path)
                        pages = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = text_splitter.split_documents(pages)
                        self.vector_store = FAISS.from_documents(
                            chunks,
                            self.embedding_model
                        )
                        self.vector_store.save_local(vector_store_path)
                        st.success("Vector store created and saved.")
                    else:
                        st.warning(f"PDF file {self.pdf_path} not found. Using empty knowledge base.")
                        # Create empty vector store
                        from langchain.schema import Document
                        empty_docs = [Document(page_content="No documents loaded")]
                        self.vector_store = FAISS.from_documents(empty_docs, self.embedding_model)
            except Exception as e:
                st.error(f"Vector store error: {str(e)}")
                # Create empty vector store as fallback
                from langchain.schema import Document
                empty_docs = [Document(page_content="Error loading documents")]
                self.vector_store = FAISS.from_documents(empty_docs, self.embedding_model)

            # Retriever
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

            # Prompt
            self.prompt = PromptTemplate.from_template(INSTRUCTIONS)

            # Memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=False
            )
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def ask_question(self, question):
        try:
            # Run query
            result = self.qa_chain.invoke({"query": question})
            # Save conversation in memory
            self.memory.save_context({"question": question}, {"answer": result["result"]})
            return result["result"]
        except Exception as e:
            return f"Error processing question: {str(e)}"

# Example usage (for testing)
if __name__ == "__main__":
    bot = LegalAdvisorBot()
    question = "What is the purpose of law?"
    answer = bot.ask_question(question)
    print("Answer:", answer)
