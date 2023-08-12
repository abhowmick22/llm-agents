from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI
import os

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# initialize Hub LLM (doesn't work)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HUGGINGFACE_KEY' # https://huggingface.co/settings/tokens
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

# initialize OpenAI
# os.environ['OPENAI_API_KEY'] = 'OPENAI_KEY'
davinci = OpenAI(model_name='text-davinci-003')

# user questions, asking one at a time
question = "Which NFL team won the Super Bowl in the 2010 season?"
questions = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.generate(questions))

# user questions, ask all in one prompt
multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""

long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?\n" +
    "How many eyes does a blade of grass have?"
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=long_prompt,
    llm=davinci
)

print(llm_chain.run(qs_str))