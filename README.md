# RAG Techniques
Exploring various way to implement RAG

## Techniques
1. [Naive RAG (Simple RAG)](./simple_rag.py)
   - Basic retrieval and generation without advanced processing

   Sample Output
   ```
   Query: Earnings before taxes
   Response: Apple Inc. reported income before provision for income taxes of $25,494 million for the
   three months ended June 29, 2024, and $93,875 million for the nine months ended June 29, 2024.
   ```

2. [Simple RAG with Memory](./rag_with_mem.py)
   - Maintains context from previous interactions

   Sample Output
   ```
   Query 1: Total marketable securities on June 29, 2024? 
   Response 1: Based on the condensed consolidated balance sheet as of June 29, 2024, the total marketable
   securities held by Apple Inc. was $127,476 million, which includes $36,236 million in current marketable
   securities and $91,240 million in non-current marketable securities.

   Query 2: How about on September 2023? 
   Response 2: According to the condensed consolidated balance sheet, as of September 30, 2023, Apple Inc.
   held total marketable securities of $132,134 million, consisting of $31,590 million in current marketable
   securities and $100,544 million in non-current marketable securities. 

   Query 3: What's the percentage change between them? 
   Response 2: To calculate the percentage change in total marketable securities between June 29, 2024 and
   September 30, 2023,
   I will use the following formula:

   Percentage change = (New Value - Original Value) / Original Value x 100

   Total marketable securities on June 29, 2024: $127,476 million
   Total marketable securities on September 30, 2023: $132,134 million

   Percentage change = ($127,476 million - $132,134 million) / $132,134 million x 100
            = (-$4,658 million) / $132,134 million x 100
            = -3.53%

   Therefore, the total marketable securities held by Apple Inc. decreased by 3.53% from September 30, 2023 to
   June 29, 2024.

   ```

3. [Branched RAG](./branched_rag.py)
   - Performs multiple retrieval steps, refining the search based on intermediate results
   
   Sample Output
   ```
   Query: Total earnings of Apple during Q3 2024 and the average stock price during that period, 
   Response: Based on the provided information, Apple Inc.'s total earnings (net income) during Q3 2024 were
   $19.8 billion. The average stock price of Apple during Q3 2024, calculated from the monthly prices in July,
   August, and September, was $232.67.
   ```

4. [Contextual RAG](./contextual_rag.py)
   - Prepends chunk-specific explanatory context to each chunk before embedding (“Contextual Embeddings”)

   Sample Output
   ```
   Query-1 : What is this paper all about?, 
   Contextual Response-1: This paper describes the design and implementation of Dynamo, a highly available key-value storage system used by Amazon for managing state across many of its core services. It explains how Dynamo utilizes consistent hashing with virtual nodes for partitioning data across nodes, and employs a replication strategy to achieve high availability and durability of data., 
   Non-contextual Response-1: This paper presents Dynamo, a highly available and scalable distributed data storage system designed by Amazon to meet the requirements of its e-commerce platform. Dynamo provides a simple key-value data model, eventual consistency, and a decentralized architecture that enables seamless scalability and high availability. The paper discusses the design considerations, implementation details, and real-world experiences of running Dynamo in production to handle massive workloads during peak shopping seasons. It highlights how different techniques were combined to achieve a highly available and performant storage system that can meet strict performance demands while allowing services to trade off between consistency, durability, and cost-effectiveness based on their specific needs.

   Query-2 : What are some of the technical considerations made in the paper?, 
   Contextual Response-2: Based on the context provided, some of the key technical considerations discussed in the paper are the tradeoffs between strong consistency and high availability in the design of Dynamo, Amazon's highly available key-value storage system. The paper likely explores how Dynamo handles network partitions and server failures while maintaining high availability, potentially sacrificing strong consistency guarantees in favor of availability. The design aims to provide a robust and resilient distributed storage solution capable of managing state across Amazon's core services., 
   Non-contextual Response-2: Some of the key technical considerations made in the paper are:

    1. Providing high availability and durability by using data replication across multiple nodes, even in the face of node failures and network partitions.

    2. Achieving eventual consistency rather than strong consistency, to favor availability over strict consistency guarantees.

    3. Using a simple key-value data model with get/put operations, avoiding complex relational schemas.

    4. Decentralized architecture with gossip-based protocols for failure detection and membership management, avoiding central coordination points.

    5. Ability to add and remove storage nodes dynamically without manual partitioning or redistribution.

    6. Expressing performance SLAs in terms of the 99.9th percentile latency rather than averages, to ensure good experience for all customers.

    7. Allowing services to trade off consistency, durability and performance based on their specific needs through tunable system properties.
   ```

4. HyDE (Hypothetical Document Embedding)
   - Generates a hypothetical ideal document before retrieval to improve search relevance

5. Adaptive RAG
   - Dynamically adjusts retrieval and generation strategies based on query type or difficulty

6. Corrective RAG (CRAG)
   - Iteratively refines generated responses by fact-checking against retrieved information

7. Self-RAG
   - Includes self-reflection and self-grading on both retrieved documents and generated responses

8. Agentic RAG
   - Combines RAG with agentic behavior for complex, multi-step problem-solving

9. Modular RAG
   - Separates retrieval and generation components into distinct, modular parts

10. Hierarchical Index Retrieval
    - Organizes data into a hierarchical structure for more precise retrieval

11. Hybrid Search
    - Integrates various search techniques, including keyword-based, semantic, and vector searches

12. Recursive Retrieval and Query Engine
    - Acquires smaller chunks initially, then larger chunks with more contextual information

13. StepBack Approach
    - Encourages reasoning around broader concepts and principles

14. Sub-Queries
    - Employs various query strategies like tree queries, vector queries, or sequential querying of chunks

15. Retriever Ensembling and Reranking
    - Combines multiple retrieval models and refines results based on additional criteria

## Dependencies
- [Chroma](https://github.com/chroma-core/chroma)
- Google GenerativeAI: Using Gemnini as the model for embeddings
- llamaindex

## High Level Workflows
```mermaid
graph LR;
    subgraph "Ingestion"
        direction LR
        Documents-->Chunks-->Embeddings-->Index;
    end
```
```mermaid
graph LR;
    subgraph Retrieval
        direction LR
        Query-->index-->top-k
    end
    
    subgraph Synthesis
        direction LR
        LLM-->Response
    end

    Retrieval-->Synthesis
```




