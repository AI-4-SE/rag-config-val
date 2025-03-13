Artifact releas for the paper: "On Automating Configuration Dependency Validation via Retrieval-Augmented Generation"

## Paper
PDF: will be linked later

## ABSTRACT</h3>

Configuration dependencies arise when multiple technologies within a software system require coordinated settings for correct interplay. Existing approaches for detecting such dependencies often suffer from high false-positive rates, require additional validation mechanisms, and are typically limited to specific projects or technologies. Recent work that incorporates large language models (LLMs) for dependency validation still suffers from inaccuracies due to project- and technology-specific variations, as well as from missing contextual information. A promising solution to missing contextual information represents retrieval-augmented generation (RAG) systems, which can dynamically retrieve project- and technology-specific knowledge for validating configuration dependencies. However, it is unclear which architectural decisions, which information resources, and what kind of information help best to automate the validation of configuration dependencies. 

In this work, we set out to evaluate whether RAG can improve LLM-based validation of configuration dependencies and which architectural decisions, as well as contextual information are needed to overcome the static knowledge base of LLMs. To this end, we conducted a large empirical study on validating configuration dependencies. Our evaluation of six state-of-the-art LLMs and eight RAG variants shows that vanilla LLMs already demonstrate solid validation abilities, while RAG has only marginal or even negative effects on the validation performance of the models. By incorporating tailored contextual information into the RAG system--derived from a qualitative analysis of validation failures--we achieve significantly more accurate validation results across all models, with an average precision of 0.84 and recall of 0.70, representing improvements of 35% and 133% over vanilla LLMs, respectively. In addition, these results offer two important insights: Simplistic RAG systems may not benefit from additional information if it is not tailored to the task at hand, and it is often unclear upfront what kind of information yields improved performance.

## Project Structure

- `/config`: contains the configuration files for the different RAG variants and for ingestion
- `/data`: contains data of subject systems, dependency datasets, ingested data, and evaluation results 
- `/evaluation`: contains script for evaluation
- `/src`: contains implementation of the RAG system

## Supported Models

<details>
<table>
  <thead>
    <tr>
      <th>Alias</th>
      <th>Model Name</th>
      <th># Params</th>
      <th>Context Length</th>
      <th>Open Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4o</td>
      <td>gpt-4o-2024-11-20</td>
      <td style="text-align: right;">-</td>
      <td style="text-align: right;">128k</td>
      <td style="text-align: right;">no</td>
    </tr>
    <tr>
      <td>4o-mini</td>
      <td>gpt-4o-mini-2024-07-18</td>
      <td style="text-align: right;">-</td>
      <td style="text-align: right;">128k</td>
      <td style="text-align: right;">no</td>
    </tr>
    <tr>
      <td>DSr:70b</td>
      <td>deepseek-r1:70b</td>
      <td style="text-align: right;">70B</td>
      <td style="text-align: right;">131k</td>
      <td style="text-align: right;">yes</td>
    </tr>
    <tr>
      <td>DSr:14b</td>
      <td>deepseek-r1:14b</td>
      <td style="text-align: right;">14B</td>
      <td style="text-align: right;">131k</td>
      <td style="text-align: right;">yes</td>
    </tr>
    <tr>
      <td>L3.1:70b</td>
      <td>llama3.1:70b</td>
      <td style="text-align: right;">70B</td>
      <td style="text-align: right;">8k</td>
      <td style="text-align: right;">yes</td>
    </tr>
    <tr>
      <td>L3.1:8b</td>
      <td>llama3.1:8b</td>
      <td style="text-align: right;">8B</td>
      <td style="text-align: right;">8k</td>
      <td style="text-align: right;">yes</td>
    </tr>
  </tbody>
</table>
</details>
</details>

## Supported RAG Variants

<details>

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Embedding Model</th>
      <th>Embedding Dimension</th>
      <th>Reranking</th>
      <th>Top N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>R1</td>
      <td>text-embed-ada-002</td>
      <td style="text-align: right;">1536</td>
      <td style="text-align: right;">Sentence Transformer</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>R2</td>
      <td>text-embed-ada-002</td>
      <td style="text-align: right;">1536</td>
      <td style="text-align: right;">Sentence Transformer</td>
      <td style="text-align: right;">3</td>
    </tr>
    <tr>
      <td>R3</td>
      <td>text-embed-ada-002</td>
      <td style="text-align: right;">1536</td>
      <td style="text-align: right;">Colbert Rerank</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>R4</td>
      <td>text-embed-ada-002</td>
      <td style="text-align: right;">1536</td>
      <td style="text-align: right;">Colbert Rerank</td>
      <td style="text-align: right;">3</td>
    </tr>
    <tr>
      <td>R5</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td style="text-align: right;">Sentence Transformer</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>R6</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td style="text-align: right;">Sentence Transformer</td>
      <td style="text-align: right;">3</td>
    </tr>
    <tr>
      <td>R7</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td style="text-align: right;">Colbert Rerank</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>R8</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td style="text-align: right;">Colbert Rerank</td>
      <td style="text-align: right;">3</td>
    </tr>
  </tbody>
</table>
</details>

## Experiments

<details>
To run the experiments on the validation effectiveness of vanilla LLMs and different RAG variants, you need to execute the ingestion once and the retrieval, and generation pipeline one after the other for a given RAG variant. Next, we describe the different steps in detail:

1. Create a ``.env`` file in the root directory containing the API token for OpenAI, Pinecone, and GitHub.

    ```
    OPENAI_KEY=<your-openai-key>
    PINECONE_API_KEY=<your-pinecone-key>
    GITHUB_TOKEN=<your-github-key>   
    ```

2. Run the ingestion pipeline once to create the specific Pinecone indices for the static and dynamic context information and already ingest the static context using the following command:

    ```python
    python ingestion_pipeline.py
    ```
    
    By default, this script uses the `.env` file in the root directory and the `ingestion.toml` in the configs directory, but they can be changed using the the corresponding command line argumengts `--config_file` and `--env_file`. The `ingestion.toml` specifies the static and dynamic indices according to the underlying embedding models and their embedding models as well as the sources of the static context, which is directly ingested after the creation of the static indices. 

3. Once the vector database is set up properly, we can start the retrieval pipeline for a given RAG variants using the following command:

    ```python
    python retrieval_pipeline.py --config_file=configs/config_{ID}.toml
    ```

    The `config_{ID}.toml` defines a specific RAG variant. The RAG variants have the IDs from 1 to 8 (R1-R8), while vanille LLMs have the ID 0. Each configuration file for a RAG variant contains the following parameters:
     - `index_name`: the index from which data should be retrieved
     - `embedding_model`: the embedding model
     - `embedding_dimension`: the dimension of the embedding model
     - `rerank`: the re-ranking algorithm
     - `top_n`: the number of chunk provided to the LLM
     - `num_websited`: number of websites to get dynamic context
     - `alpha`: the weight for sparse/dense retrieval
     - `web_search_enabled`: defined whether Web search is enabled or not
     - `inference_models`: list of LLMs for generation
     - `temperature`: temperature of LLMs
     - `data_file`: path of data file containing the dependencies for validation
     - `retrieval_file`: path of data file in which the retrieval results should be stored
     - `generation_file`: path of data file in which the generation results should be stores.
     

    This script iterates through all dependencies, retrieves static and dynamic context, and finally stores the retrieval results.


4.  Once the additional context is retrieved, we can run the generation pipeline with the following command:
    
    ```python
        python generation_pipeline.py --config_file=configs/config_{ID}.toml
    ```

    This script takes as input the same configuration file that we use for running the retrieval pipeline. For each inference model specified, it iterates through all dependencies, validates them with the additional context, and finally stores the generation results.


5. To run the retrieval and generation for the refined vanilla LLMs and refined RAG variant, execute step 2 and 3 with the corresponding configuration file: `configs/advanced_{ID}`, where ID can either be 0 for refined vanille LLMs or 1 for refined RAG variant R1.


6. To compute the validation effectiveness of a vanilla LLMs or a specific RAG variant switch to the `evaluation` directory and execute the following command:
    
    ```python
    python metrics.py --genration_file={generation_file}.json
    ```

</details>

