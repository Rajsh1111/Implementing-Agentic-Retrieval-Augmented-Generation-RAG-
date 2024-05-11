#References:LlamaIndex V 0.10.15 Documentation , AI User Conference 2024 : Jerry Liu Keynote


#Code Implementation

#Install required Dependencies
!pip install -qU langchain langchain_openai langgraph arxiv duckduckgo-search
!pip install -qU faiss-cpu pymupdf

#Set up Environmental Varaibales
from google.colab import userdata
from uuid import uuid4
import os
#
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - LangGraph - {uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] =  userdata.get('LANGCHAIN_API_KEY')

#Instantiate a Simple Retrieval Chain using LCEL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load the document pertaining to a particular topic
docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5).load()

# Split the dpocument into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=350, chunk_overlap=50
)

chunked_documents = text_splitter.split_documents(docs)
#
# Instantiate the Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=os.environ['OPENAI_API_KEY'])
# Create Index- Load document chunks into the vectorstore
faiss_vectorstore = FAISS.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
)
# Create a retriver
retriever = faiss_vectorstore.as_retriever()

#Generate RAG prompt
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """\
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

#Instantiate the LLM

from langchain_openai import ChatOpenAI

openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")

#Build LCEL RAG Chain
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retrieval_augmented_generation_chain = (
       {"context": itemgetter("question") 
    | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)
#
retrieval_augmented_generation_chain

################### RESPONSE ###############################

  context: RunnableLambda(itemgetter('question'))
           | VectorStoreRetriever(tags=['FAISS', 'OpenAIEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x7de64bd18f40>),
  question: RunnableLambda(itemgetter('question'))
}
| RunnableAssign(mapper={
    context: RunnableLambda(itemgetter('context'))
  })
| {
    response: ChatPromptTemplate(input_variables=['context', 'question'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.\n\nQuestion:\n{question}\n\nContext:\n{context}\n"))])
              | ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x7de64d033850>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x7de64d032b60>, openai_api_key=SecretStr('**********'), openai_proxy=''),
    context: RunnableLambda(itemgetter('context'))
	
#Ask Query

await retrieval_augmented_generation_chain.ainvoke({"question" : "What is Retrieval Augmented Generation?"})

##################### REPONSE #############################
{'response': AIMessage(content='Retrieval Augmented Generation is a text generation paradigm that combines deep learning technology with traditional retrieval technology. It has achieved state-of-the-art performance in many NLP tasks by explicitly acquiring knowledge in a plug-and-play manner, leading to scalability and potentially alleviating the difficulty of text generation. It involves generating text from retrieved human-written references rather than generating from scratch.', response_metadata={'token_usage': {'completion_tokens': 73, 'prompt_tokens': 2186, 'total_tokens': 2259}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'stop', 'logprobs': None}),
 'context': [Document(page_content='grating translation memory to NMT models (Gu\net al., 2018; Zhang et al., 2018; Xu et al., 2020;\nHe et al., 2021). We also review the applications\nof retrieval-augmented generation in other genera-\ntion tasks such as abstractive summarization (Peng\net al., 2019), code generation (Hashimoto et al.,\n2018), paraphrase (Kazemnejad et al., 2020; Su\net al., 2021b), and knowledge-intensive generation\n(Lewis et al., 2020b). Finally, we also point out\nsome promising directions on retrieval-augmented\ngeneration to push forward the future research.\n2\nRetrieval-Augmented Paradigm\nIn this section, we ﬁrst give a general formulation\nof retrieval-augmented text generation. Then, we\ndiscuss three major components of the retrieval-\naugmented generation paradigm, including the re-\narXiv:2202.01110v2  [cs.CL]  13 Feb 2022\nInput\nSources \n(Sec. 2.2):\nTraining \nCorpus\nExternal Data\nUnsupervised \nData\nMetrics\n(Sec. 2.3):\nSparse-vector \nRetrieval\nDense-vector \nRetrieval\nTask-specific \nRetrieval\nRetrieval Memory\nGeneration Model\nSec. 4: Machine \nTranslation\nSec. 5: Other \nTasks\nData \nAugmentation\nAttention \nMechanism\nSkeleton & \nTemplates\nInformation Retrieval', metadata={'Published': '2022-02-13', 'Title': 'A Survey on Retrieval-Augmented Text Generation', 'Authors': 'Huayang Li, Yixuan Su, Deng Cai, Yan Wang, Lemao Liu', 'Summary': 'Recently, retrieval-augmented text generation attracted increasing attention\nof the computational linguistics community. Compared with conventional\ngeneration models, retrieval-augmented text generation has remarkable\nadvantages and particularly has achieved state-of-the-art performance in many\nNLP tasks. This paper aims to conduct a survey about retrieval-augmented text\ngeneration. It firstly highlights the generic paradigm of retrieval-augmented\ngeneration, and then it reviews notable approaches according to different tasks\nincluding dialogue response generation, machine translation, and other\ngeneration tasks. Finally, it points out some important directions on top of\nrecent methods to facilitate future research.'}),
  Document(page_content='augmented generation as well as three key com-\nponents under this paradigm, which are retrieval\nsources, retrieval metrics and generation models.\nThen, we introduce notable methods about\nretrieval-augmented generation, which are orga-\nnized with respect to different tasks. Speciﬁcally,\non the dialogue response generation task, exem-\nplar/template retrieval as an intermediate step has\nbeen shown beneﬁcial to informative response gen-\neration (Weston et al., 2018; Wu et al., 2019; Cai\net al., 2019a,b). In addition, there has been growing\ninterest in knowledge-grounded generation explor-\ning different forms of knowledge such as knowl-\nedge bases and external documents (Dinan et al.,\n2018; Zhou et al., 2018; Lian et al., 2019; Li et al.,\n2019; Qin et al., 2019; Wu et al., 2021; Zhang et al.,\n2021). On the machine translation task, we summa-\nrize the early work on how the retrieved sentences\n(called translation memory) are used to improve\nstatistical machine translation (SMT) (Koehn et al.,\n2003) models (Simard and Isabelle, 2009; Koehn\nand Senellart, 2010) and in particular, we inten-\nsively highlight several popular methods to inte-\ngrating translation memory to NMT models (Gu\net al., 2018; Zhang et al., 2018; Xu et al., 2020;\nHe et al., 2021). We also review the applications', metadata={'Published': '2022-02-13', 'Title': 'A Survey on Retrieval-Augmented Text Generation', 'Authors': 'Huayang Li, Yixuan Su, Deng Cai, Yan Wang, Lemao Liu', 'Summary': 'Recently, retrieval-augmented text generation attracted increasing attention\nof the computational linguistics community. Compared with conventional\ngeneration models, retrieval-augmented text generation has remarkable\nadvantages and particularly has achieved state-of-the-art performance in many\nNLP tasks. This paper aims to conduct a survey about retrieval-augmented text\ngeneration. It firstly highlights the generic paradigm of retrieval-augmented\ngeneration, and then it reviews notable approaches according to different tasks\nincluding dialogue response generation, machine translation, and other\ngeneration tasks. Finally, it points out some important directions on top of\nrecent methods to facilitate future research.'}),
  Document(page_content='recent methods to facilitate future research.\n1\nIntroduction\nRetrieval-augmented text generation, as a new\ntext generation paradigm that fuses emerging deep\nlearning technology and traditional retrieval tech-\nnology, has achieved state-of-the-art (SOTA) per-\nformance in many NLP tasks and attracted the at-\ntention of the computational linguistics community\n(Weston et al., 2018; Dinan et al., 2018; Cai et al.,\n2021). Compared with generation-based counter-\npart, this new paradigm has some remarkable ad-\nvantages: 1) The knowledge is not necessary to be\nimplicitly stored in model parameters, but is explic-\nitly acquired in a plug-and-play manner, leading\nto great scalibility; 2) Instead of generating from\nscratch, the paradigm generating text from some re-\ntrieved human-written reference, which potentially\nalleviates the difﬁculty of text generation.\nThis paper aims to review many representative\napproaches for retrieval-augmented text generation\ntasks including dialogue response generation (We-\nston et al., 2018), machine translation (Gu et al.,\n2018) and others (Hashimoto et al., 2018). We\n∗All authors contributed equally.\nﬁrstly present the generic paradigm of retrieval-\naugmented generation as well as three key com-\nponents under this paradigm, which are retrieval\nsources, retrieval metrics and generation models.\nThen, we introduce notable methods about', metadata={'Published': '2022-02-13', 'Title': 'A Survey on Retrieval-Augmented Text Generation', 'Authors': 'Huayang Li, Yixuan Su, Deng Cai, Yan Wang, Lemao Liu', 'Summary': 'Recently, retrieval-augmented text generation attracted increasing attention\nof the computational linguistics community. Compared with conventional\ngeneration models, retrieval-augmented text generation has remarkable\nadvantages and particularly has achieved state-of-the-art performance in many\nNLP tasks. This paper aims to conduct a survey about retrieval-augmented text\ngeneration. It firstly highlights the generic paradigm of retrieval-augmented\ngeneration, and then it reviews notable approaches according to different tasks\nincluding dialogue response generation, machine translation, and other\ngeneration tasks. Finally, it points out some important directions on top of\nrecent methods to facilitate future research.'}),
  Document(page_content='A Survey on Retrieval-Augmented Text Generation\nHuayang Li♥,∗\nYixuan Su♠,∗\nDeng Cai♦,∗\nYan Wang♣,∗\nLemao Liu♣,∗\n♥Nara Institute of Science and Technology\n♠University of Cambridge\n♦The Chinese University of Hong Kong\n♣Tencent AI Lab\nli.huayang.lh6@is.naist.jp, ys484@cam.ac.uk\nthisisjcykcd@gmail.com, brandenwang@tencent.com\nlemaoliu@gmail.com\nAbstract\nRecently, retrieval-augmented text generation\nattracted increasing attention of the compu-\ntational linguistics community.\nCompared\nwith conventional generation models, retrieval-\naugmented text generation has remarkable ad-\nvantages and particularly has achieved state-of-\nthe-art performance in many NLP tasks. This\npaper aims to conduct a survey about retrieval-\naugmented text generation. It ﬁrstly highlights\nthe generic paradigm of retrieval-augmented\ngeneration, and then it reviews notable ap-\nproaches according to different tasks including\ndialogue response generation, machine trans-\nlation, and other generation tasks. Finally, it\npoints out some promising directions on top of\nrecent methods to facilitate future research.\n1\nIntroduction\nRetrieval-augmented text generation, as a new\ntext generation paradigm that fuses emerging deep\nlearning technology and traditional retrieval tech-', metadata={'Published': '2022-02-13', 'Title': 'A Survey on Retrieval-Augmented Text Generation', 'Authors': 'Huayang Li, Yixuan Su, Deng Cai, Yan Wang, Lemao Liu', 'Summary': 'Recently, retrieval-augmented text generation attracted increasing attention\nof the computational linguistics community. Compared with conventional\ngeneration models, retrieval-augmented text generation has remarkable\nadvantages and particularly has achieved state-of-the-art performance in many\nNLP tasks. This paper aims to conduct a survey about retrieval-augmented text\ngeneration. It firstly highlights the generic paradigm of retrieval-augmented\ngeneration, and then it reviews notable approaches according to different tasks\nincluding dialogue response generation, machine translation, and other\ngeneration tasks. Finally, it points out some important directions on top of\nrecent methods to facilitate future research.'})]}

#Creating our Tool Belt

#As is usually the case, we’ll want to equip our agent with a toolbelt to help answer questions and add external knowledge.
#There’s a load of tools in the LangChain Community Repo but we’ll stick to a couple just so we can observe the cyclic nature of LangGraph in action!

#Here we will leverage:

#Duck Duck Go Web Search

#Arxiv
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.prebuilt import ToolExecutor
tool_belt = [
    DuckDuckGoSearchRun(),
    ArxivQueryRun()
]

tool_executor = ToolExecutor(tool_belt)

#Instantiate Openai Function Calling

from langchain_core.utils.function_calling import convert_to_openai_function
#
model = ChatOpenAI(temperature=0)
#
functions = [convert_to_openai_function(t) for t in tool_belt]
model = model.bind_functions(functions)

#Leverage LangGraph

#LangGraph leverages a StatefulGraph which uses an AgentState object to pass information between the various nodes of the graph.

#There are more options than what we’ll see below — but this AgentState object is one that is stored in a TypedDict with the key messages and the value is a Sequence of BaseMessages that will be appended to whenever the state changes.

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]
  
 #Build Nodes

#call_model is a node that will…well…call the model
#call_tool is a node which will call a tool

from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage

def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {"messages" : [response]}

def call_tool(state):
  last_message = state["messages"][-1]

  action = ToolInvocation(
      tool=last_message.additional_kwargs["function_call"]["name"],
      tool_input=json.loads(
          last_message.additional_kwargs["function_call"]["arguments"]
      )
  )

  response = tool_executor.invoke(action)

  function_message = FunctionMessage(content=str(response), name=action.tool)

  return {"messages" : [function_message]}
  
 #Build the Workflow
 
 from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.nodes
########################RESPONSE############################
{'agent': RunnableLambda(call_model), 'action': RunnableLambda(call_tool)}

#Set the Entry point
workflow.set_entry_point("agent")

#Build a Conditional Edge for Routing

def should_continue(state):
  last_message = state["messages"][-1]

  if "function_call" not in last_message.additional_kwargs:
    return "end"

  return "continue"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue" : "action",
        "end" : END
    }
)

#Finally connect the conditional edge to the agent node and action node

workflow.add_edge("action", "agent")

#Compile the workflow

app = workflow.compile()
#
app
#
################### RESPONSE ###############################
CompiledGraph(nodes={'agent': ChannelInvoke(bound=RunnableLambda(call_model)
| ChannelWrite(channels=[ChannelWriteEntry(channel='agent', value=None, skip_none=False), ChannelWriteEntry(channel='messages', value=RunnableLambda(...), skip_none=False)]), config={'tags': []}, channels={'messages': 'messages'}, triggers=['agent:inbox'], mapper=functools.partial(<function _coerce_state at 0x7de64d4c9ab0>, <class '__main__.AgentState'>)), 'action': ChannelInvoke(bound=RunnableLambda(call_tool)
| ChannelWrite(channels=[ChannelWriteEntry(channel='action', value=None, skip_none=False), ChannelWriteEntry(channel='messages', value=RunnableLambda(...), skip_none=False)]), config={'tags': []}, channels={'messages': 'messages'}, triggers=['action:inbox'], mapper=functools.partial(<function _coerce_state at 0x7de64d4c9ab0>, <class '__main__.AgentState'>)), 'agent:edges': ChannelInvoke(bound=RunnableLambda(runnable), config={'tags': ['langsmith:hidden']}, channels={'messages': 'messages'}, triggers=['agent']), 'action:edges': ChannelInvoke(bound=ChannelWrite(channels=[ChannelWriteEntry(channel='agent:inbox', value='action', skip_none=True)]), config={'tags': ['langsmith:hidden']}, channels={'messages': 'messages'}, triggers=['action']), '__start__': ChannelInvoke(bound=ChannelWrite(channels=[ChannelWriteEntry(channel='__start__', value=None, skip_none=False), ChannelWriteEntry(channel='messages', value=RunnableLambda(...), skip_none=False)]), config={'tags': ['langsmith:hidden']}, channels={None: '__start__:inbox'}, triggers=['__start__:inbox']), '__start__:edges': ChannelInvoke(bound=ChannelWrite(channels=[ChannelWriteEntry(channel='agent:inbox', value=None, skip_none=False)]), config={'tags': ['langsmith:hidden']}, channels={'messages': 'messages'}, triggers=['__start__'])}, channels={'messages': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x7de64de19e70>, 'agent:inbox': <langgraph.channels.any_value.AnyValue object at 0x7de64de19fc0>, 'action:inbox': <langgraph.channels.any_value.AnyValue object at 0x7de64de1a080>, '__start__:inbox': <langgraph.channels.any_value.AnyValue object at 0x7de64de1ad10>, 'agent': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7de64de1b2b0>, 'action': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7de64de1b880>, '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7de64de18340>, '__end__': <langgraph.channels.last_value.LastValue object at 0x7de64de18490>, <ReservedChannels.is_last_step: 'is_last_step'>: <langgraph.channels.last_value.LastValue object at 0x7de64de1b670>}, output='__end__', hidden=['agent:inbox', 'action:inbox', '__start__', 'messages'], snapshot_channels=['messages'], input='__start__:inbox', graph=<langgraph.graph.state.StateGraph object at 0x7de64de18880>)

#Invoke the LangGraph- Ask Query

from langchain_core.messages import HumanMessage

inputs = {"messages" : [HumanMessage(content="What is RAG in the context of Large Language Models? When did it break onto the scene?")]}

response = app.invoke(inputs)
print(response)
############################RESPOSNE #########################
{'messages': [HumanMessage(content='What is RAG in the context of Large Language Models? When did it break onto the scene?'),
  AIMessage(content='', additional_kwargs={'function_call': {'arguments': '{"query":"RAG in the context of Large Language Models"}', 'name': 'duckduckgo_search'}}, response_metadata={'token_usage': {'completion_tokens': 25, 'prompt_tokens': 171, 'total_tokens': 196}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'function_call', 'logprobs': None}),
  FunctionMessage(content="Large language models (LLMs) are incredibly powerful tools for processing and generating text. However, they inherently struggle to understand the broader context of information, especially when dealing with lengthy conversations or complex tasks. This is where large context windows and Retrieval-Augmented Generation (RAG) come into play. These advanced, generalized language models are trained on vast datasets, enabling them to understand and generate human-like text. In the context of RAG, LLMs are used to generate fully formed responses based on the user query and contextual information retrieved from the vector DBs during user queries. Querying In the rapidly evolving landscape of language models, the debate between Retrieval-Augmented Generation (RAG) and Long Context Large Language Models (LLMs) has garnered significant attention. Retrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information. Implementing RAG in an LLM-based question answering system has two main benefits: It ensures that the model has ... RAG stands for R etrieval- A ugmented G eneration. RAG enables large language models (LLM) to access and utilize up-to-date information. Hence, it improves the quality of relevance of the response from LLM. Below is a simple diagram of how RAG is implemented.", name='duckduckgo_search'),
  AIMessage(content="RAG stands for Retrieval-Augmented Generation in the context of Large Language Models (LLMs). It is an AI framework that improves the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information. RAG enables LLMs to access and utilize up-to-date information, thereby improving the relevance and quality of the responses generated by the model. RAG broke onto the scene in the rapidly evolving landscape of language models as a way to enhance the capabilities of LLMs in understanding and generating human-like text.", response_metadata={'token_usage': {'completion_tokens': 117, 'prompt_tokens': 491, 'total_tokens': 608}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None})]}
  
  response['messages'][-1].content

####### RESPONSE #####################
RAG stands for Retrieval-Augmented Generation in the context of Large Language Models (LLMs). It is an AI framework that improves the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information. RAG enables LLMs to access and utilize up-to-date information, thereby improving the relevance and quality of the responses generated by the model. RAG broke onto the scene in the rapidly evolving landscape of language models as a way to enhance the capabilities of LLMs in understanding and generating human-like text.

#Ask Query

question = "Who is the main author on the Retrieval Augmented Generation paper - and what University did they attend?"

inputs = {"messages" : [HumanMessage(content=question)]}
#
response = app.invoke(inputs)
print(response['messages'][-1].content)
################# RESPONSE ##############################
The main author on the "Retrieval Augmented Generation" paper is Huayang Li. Unfortunately, the University they attended is not mentioned in the summary provided.

#Ask Query

question = "Who is the main author on the Retrieval Augmented Generation paper?"

inputs = {"messages" : [HumanMessage(content=question)]}

response = app.invoke(inputs)
print(response['messages'][-1].content)
################# RESPONSE ##############################
The main authors on the paper "A Survey on Retrieval-Augmented Text Generation" are Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu.

