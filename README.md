# Movie Recommendation RAG
---

## Overview

The **Movie Recommendation RAG** is an AI-powered chatbot designed to generate movie recommendations for an imaginary OTT (Over-The-Top) platform. Leveraging the power of Llama-3 as the generation model and databricks-bge-large-en for embedding, this application aims to provide personalized movie suggestions based on user preferences.

## Features

- **Recommendations**: The chatbot analyzes user input and suggests relevant movies based on available labels within the platform.
- **Hallucination**: As part of its learning process, the chatbot can generate creative recommendations beyond existing data.
- **Hackathon Project**: Developed for a hackathon, this application is not intended for production use.

## Usage

1. **Input**: Users can query using keywords, genres, or specific movie titles.
2. **Output**: The chatbot responds with a list of recommended movies.

## Resources Used:

1. Databricks:
    - Unity Catalog
    - Model Serving
    - Vector Search
    - CLI
    - Service Principal
    - Run time : 13.3 LTS ML (includes Apache Spark 3.4.1, Scala 2.12)
2. Hugging Face:
    - Spaces
    - UI built using Gradio
3. LangChain
4. MLFlow

## Datasets Citation:

1. > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=<http://dx.doi.org/10.1145/2827872>

2. > Free IMDB data set from Bright Data=<https://brightdata.com/products/datasets/marketplace#all> 

## Installation

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Upload the above datasets into a Databricks Unity Catalog volumes.
4. Open the notebooks in Data Ingestion folder, update the variables & run both the notebooks.
5. Open the notebooks in Data Preprocessing folder, update the variables & run the notebook.
6. On the final movies_info table resulting from step 6, create a vector search index. 
    - Create a vector search end point first by heading onto databricks compute section.
    - Then using the above vector search endpoint you can create the vecor index.
    - select readily available _databricks-bge-large-en_ as the embedding model.
7. Create a Service Principle & set it up as a profile for Databricks CLI in .databrickscfg file in local machine [link](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/authentication#azure-sp-auth)
    - Create a secret scope **_arm_spn_mv_secrets_** using the profile
      `databricks secrets create-scope arm_spn_mv_secrets --profile <profile name>`
    - Create a PAT token on behalf of the service principal using 
      `databricks tokens create --comment "<comment>" --lifetime-seconds <3600> -p <profile name>`
    - Add the service principal client_id **_spn_client_id_** & the PAT token **_spn_dbx_secret_** to the secret scope
      `databricks secrets put-secret arm_spn_mv_secrets <secret-name> --profile <profile name>`
8. Open the RAG Modelling notebook in Modelling folder, update the variables & run it
    - This will use _databricks-bge-large-en_ as the embedding model & _databricks-meta-llama-3-70b-instruct_ as the generation model
    - Builds a RAG chain
    - Logs & registers this RAG chain to Unity Catalog
    - Serve the latest logged model as an endpoint
    - Log the inferences to the payload inference table
9. Now, create a Hugging Face space & copy the contents of ChatBot folder into the space
    - Add the Space level secrets _API_TOKEN_ as the value of **_spn_dbx_secret_** & _API_END_POINT_ as the end point url of the served model by going into the space settings.

## Next steps:

1. Implement a feedback loop to do reinforcement learning through human feedback for providing personalized recommendations with respect to each user.
2. Implement the LLM evaluation for Hallucination, Toxicity, Relevance etc.
3. Chunk the documents & implement context & content based filtering for improving the recommendations of the generation model.
4. Secure personal user information from leaking by the LLM
5. Improve the UI 
6. Create workflows to orchestrate the flow
7. Implement Data Governance to secure the user data
8. Bundle the code through Databricks Asset Bundles

## Disclaimer

This application is purely experimental and should not be used in production environments. It may produce imaginative or unconventional recommendations. Any recommendations generated this way are reliant on the input movie records which in this use case translates to the movie labels available for hosting with the platform.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


---
