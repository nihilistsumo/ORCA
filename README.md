# ORCA: Overview Retriever with Clustering Augmentation
ORCA, in short for Overview Retriever with Clustering Augmentation is an application of the Topic-Mono-BERT model, proposed in our FIRE 2022 paper titled "Topic-Mono-BERT: A Joint Retrieval-Clustering System for Retrieving Overview Passages".

For most queries, the set of relevant documents spans multiple subtopics. Inspired by the neural ranking models and query-specific neural clustering models, we develop Topic-Mono-BERT which performs both tasks jointly. Based on text embeddings of BERT, our model learns a shared embedding that is optimized for both tasks. The clustering hypothesis would suggest that embeddings which place topically similar text in close proximity will also perform better on ranking tasks. Our model is trained with the Wikimarks approach to obtain training signals for relevance and subtopics on the same queries.
  
Our task is to identify overview passages that can be used to construct a succinct answer to the query. Our empirical evaluation on two publicly available passage retrieval datasets suggests that including the clustering supervision in the ranking model leads to about $16\%$ improvement in identifying text passages that summarize different subtopics within a query.
