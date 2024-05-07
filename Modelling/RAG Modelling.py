# Databricks notebook source
# DBTITLE 1,Install Library dependencies
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks] 
# MAGIC %pip install --upgrade sqlalchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration set up

# COMMAND ----------

catalog = "gen_ai_hackathon"
movies_db = "vector_store"
VECTOR_SEARCH_ENDPOINT_NAME = "genai_hackathon_vectorsearch"

# COMMAND ----------

movies_index_name = f"{catalog}.{movies_db}.movie_embeddings"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# url used to send the request to your model from the serverless endpoint
import os
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("arm_spn_mv_secrets", "spn_dbx_token")

# COMMAND ----------

client_id = dbutils.secrets.get("arm_spn_mv_secrets", "spn_client_id")
spark.sql(f"GRANT USAGE ON CATALOG gen_ai_hackathon TO `{client_id}`");
spark.sql(f"GRANT USAGE ON DATABASE gen_ai_hackathon.vector_store TO `{client_id}`");
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
WorkspaceClient().grants.update(c.SecurableType.TABLE, movies_index_name, 
                                changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal=dbutils.secrets.get("arm_spn_mv_secrets", "spn_client_id"))])

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG Implementation

# COMMAND ----------

# DBTITLE 1,Define the embedding retreiver
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('Horror')[:20]}...")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=movies_index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="movies_json", embedding=embedding_model
    )
    return vectorstore.as_retriever()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Generator model as llama 3

# COMMAND ----------

# Test Databricks Foundation LLM model
from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct")
print(f"Test chat model: {chat_model.predict('Annabell is very scary. I am looking for movies equally horrifiying?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the LLM chain to implemet RAG

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """You are a professional movie reviewer & recommendation assistant for movie lovers. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are unbiased and constructive in nature. You are allowed only to answer questions about movies and recommendations. If no context is available, just say that there are no movies relevant to query with us, don't try to make up an answer. Take your time to answer & keep the answer as concise as possible. Use the following pieces of context in brackets, the instructions to infer from the context are provided in the triple back ticks, follow them to answer the question in square brackets at the end:
instructions: ```
    1. The context will have a list of movie records in JSON format. 
    2. Analyse the sentiment of the user question in square brackets
    3. Give importance to the keys title, genome_tags, genres, storyline, awards, imdb_rating, review_rating, top_cast, credit, then rest of keys in the order specified.
    4. If the question references any actor/cast & crew, refer for them under the keys top_cast, credit.
    5. Take the sentiment of user question into consideration, and answer the user question by find the best movie that you can recommend the user by following the importance stated in step 3.
    6. Draft reason for each recommendation in an unbiased, & professional manner without making it up. 
    7. Provide your recommendation/recommendations title, reason for recommendation, url of the movie. Do keeping in mind to not reveal any internal information like genome_tags & scores, user personal tags, movie_id, imdb_id etc.
    8. If user requested movies are not matching with any record in the context provided, reply that 'The movies requested are not in our database. Hence, we can't fulfill your request. Apologies for the inconvenience!', don't try to make up an answer.
```
context:({context})
Question: [{question}]
Answer:
"""

prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing

# COMMAND ----------

# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"query": "I loved 'Inception'. Can you suggest similar mind-bending movies?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to UC

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{movies_db}.movie_recommendation_model_v2"

with mlflow.start_run(run_name="movie_recommendation_chatbot") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# DBTITLE 1,Function to get the latest model version registered
from mlflow.tracking import MlflowClient

def get_latest_model_version(model_name):
    client = MlflowClient()
    
    # Get the list of registered model versions
    versions = client.search_model_versions(f"name = '{model_name}'")
    
    # Sort the versions by the last_updated_timestamp in descending order
    sorted_versions = sorted(versions, key=lambda v: v.last_updated_timestamp, reverse=True)
    
    if sorted_versions:
        # Return the highest version number
        return sorted_versions[0].version
    
    # Return None if no versions are found
    return None

print(get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serving the registered model by creating a new endpoint or updating the existing endpoint

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, AutoCaptureConfigInput

serving_endpoint_name = f"movie_recommendation_engine"
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/arm_spn_mv_secrets/spn_dbx_token}}",  # <scope>/<secret> that contains an access token
            }
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name="gen_ai_hackathon",
        enabled=True, #movies_info
        schema_name="silver",
        table_name_prefix="movie_recommendation_rag"
    )
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')
