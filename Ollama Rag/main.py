from ChatPDF import ChatPDF

rag = ChatPDF()
rag.ingest('''SentenceTransformers is a Python framework for state-of-the-art sentence, text, and image embeddings. Embeddings can be computed for 100+ languages and they can be easily used for common tasks like semantic text similarity, semantic search, and paraphrase mining.

The framework is based on PyTorch and Transformers and offers a large collection of pre-trained models tuned for various tasks. Further, it is easy to fine-tune your own models.

Read the paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks for a deep dive into how the models have been trained. In this article, we'll see code examples for some of the possible use-cases of the library. Model training will be covered in a later article.''')
# rag.ingest('/Data/attention is all you need.pdf')
res = rag.ask('what is attention mechanism')
print(res)