## Advanced Retrieval Augmented Generation: Hybrid Retrieval Pipeline with LLM-Augmented Knowledge Graphs and Vector Database for Accreditation Reporting Assistance

### Abstract
We implement an advanced retrieval augmented generation pipeline with both a vector database and knowledge graph as the knowledge source used to ground responses to input queries. There are multiple knowledge graphs containing data from two distinct sources relevant to the use case. To develop our knowledge graphs we utilized both a manual construction process as well as an ‘LLM Augmented Knowledge Graph’ approach. The pipeline implements query expansion and query transformation optimization techniques through use of Open AI’s function calling for ‘multi-query’ and ‘subquery’ generation tasks. Grounding contexts are retrieved from both knowledge graph and vector index, and passed into the generator resulting in the response to the original input query. We evaluated the pipeline using the RAGAs framework and observed stable performance on answer relevancy and answer correctness metric. The pipeline is applied the use case of accreditation reporting in higher education.





![Full RAG Pipeline](diagrams/adv_rag_pipeline_cedwards.png.png)






### Project Overview:

This project implements an advanced retrieval augmented generation architecture, integrating both data stored as vector embeddings and data structured as a knowledge graph. This project implements several query optimization techniques and multi-source retrieval to provide context to ground the generated output response. This project uses Large Language Models for several tasks including knowledge graph construction. 

The project is built using Python and Cypher, with Neo4j as the database management system (DBMS). The project also combines frameworks like LangChain, with custom pipeline development. Along with the RAGAs framework for pipeline evaluation.

GIT: https://github.com/CS-Edwards/advRAG
BRANCH: Accreditation

### Introduction:

Accreditation maintenance for degree granting higher education institutions is a multi-year task involving dozens of internal and external stakeholders. For business schools nationally and internationally the Association to Advance Collegiate Schools of Business (AACSB) accreditation is the gold standard for accreditation. The accreditation signifies a commitment to: engaging curriculum, student success, valuable research contribution, cutting edge facilities and more. Over 900 business schools across the world are AACSB accredited. 

As the current protocol stands, schools are required to submit a ‘Continuous Improvement Report’ and prepare for an AACSB on-site visit every five years. The development of this report and preparation for the peer review visit can take about a year of preparation. The school’s are evaluated based on their alignment with the AACSB Standards. 

The goal of this project is to implement an advanced retrieval augmented generation pipeline, for the purposes of easing the report development process. Utilizing this pipeline, institutions will be able to integrate their own documents into the database which is prepopulated with AACSB Standard data.

Users are able to query their own institutional data and alongside AACSB data, to evaluate how their documents align with the standards. Given that the queries are based on natural language, the application makes institutional accreditation accessible to a wide range of stakeholders including, administrators, faculty, students, and committees.

