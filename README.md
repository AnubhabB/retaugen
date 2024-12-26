# Local RAG Desktop QA App

This **Experimental** project is a part of my effort at demystifying **R**etrieval-**A**ugmented **G**eneration (RAG) served as a desktop app.

## Outline:

This is a *almost* framework free, hands on implementation in Rust and a detailed and step-by-step guide to building this can be found in this [series: Desktop App for Document QA with RAG](https://blog.anubhab.me/series/desktop-app-for-document-qa-with-rag/). Here's the outline of what to expect:

- [**Part 1: Generating Embeddings**](https://blog.anubhab.me/tech/rag-desktop-app/part-1/): we implement a workflow to generate Embeddings from text data using [Stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5) and some context-aware text-splitting techniques using the crate `text-splitter`.
- [**Part 2: Vector Storage**](https://blog.anubhab.me/tech/rag-desktop-app/part-2/) we’ll build our own mini Vector Store inspired by [Spotify’s ANNOY](https://github.com/spotify/annoy).
- [**Part 3: Document Layout Analysis, Text Extraction and Generation**](https://blog.anubhab.me/tech/rag-desktop-app/part-3/): we code up a pipeline to analyze and extract text from `.pdf` files and also set the foundation for text generation with a LLaMA Model.
- [**Part 4: Indexing and Search](https://blog.anubhab.me/tech/rag-desktop-app/part-4/): we work on the `retrieve-and-answer` flow from our corpus.
- [**Part 5: Techniques**](https://blog.anubhab.me/tech/rag-desktop-app/part-5/): we implement and evaluate some techniques for a better RAG.

## Output:


**This video has been sped up**