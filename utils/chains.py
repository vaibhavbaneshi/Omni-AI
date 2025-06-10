from common.langchain_imports import LLMMathChain

from utils.llm import llm

math_chain=LLMMathChain.from_llm(llm=llm)