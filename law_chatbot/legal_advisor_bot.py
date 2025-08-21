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
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    def __init__(self, pdf_path="law_chatbot/law1.pdf", persist_directory="law_chatbot/chroma_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
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
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=None,
                proxies=None,
                api_key=GROQ_API_KEY
            )
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in environment or Streamlit secrets.")

            # Embeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            # Load existing vector store
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name="legal_collection"
                )
                print("Loaded existing vector store.")
            else:
                if os.path.exists(self.pdf_path):
                    loader = PyPDFLoader(self.pdf_path)
                    pages = loader.load()
                    print(f"Loaded {len(pages)} pages from PDF.")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(pages)
                    print(f"Split into {len(chunks)} chunks.")
                    self.vector_store = Chroma.from_documents(
                        chunks,
                        self.embedding_model,
                        persist_directory=self.persist_directory,
                        collection_name="legal_collection"
                    )
                    print("Vector store created and persisted.")
                else:
                    raise FileNotFoundError(f"Neither {self.persist_directory} nor {self.pdf_path} found. Please ensure they are included.")

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
                chain_type_kwargs={"prompt": self.prompt}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def ask_question(self, question):
        # Run query
        result = self.qa_chain.invoke({"query": question})
        # Save conversation in memory
        self.memory.save_context({"question": question}, {"answer": result["result"]})
        return result["result"]

# Example usage (for testing)
if __name__ == "__main__":
    bot = LegalAdvisorBot()
    question = "What is the purpose of law?"
    answer = bot.ask_question(question)
    print("Answer:", answer)
