from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# initialize Hub LLM
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

print("here")
# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

# ask the user question about NFL 2010
print(llm_chain.run(question))
