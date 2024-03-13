import os
import uuid
from ragas.testset  import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
load_dotenv()
unique_id = uuid.uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
loader = DirectoryLoader("./docs")
documents = loader.load()
print(documents)

# generator with openai models
generator = TestsetGenerator.with_openai(critic_llm="gpt-3.5-turbo-1106",generator_llm="gpt-3.5-turbo-1106")

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=1, distributions={simple: 1})
testset.to_pandas()