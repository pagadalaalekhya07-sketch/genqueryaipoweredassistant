from studentapp.rag.embeddings import embed_texts
from studentapp.rag.generater import generate_answer
from studentapp.rag.vector_store import serch_chunks
from studentapp.rag.structured_retriver import (
    get_student_data,
    student_data_to_context,
)


# Main RAG pipeline
# Connects student data, document retrieval,
# and Gemini to generate the final answer.
def ask_rag(question, user):
    # Get the logged-in student's structured data
    student_data = get_student_data(user)

    structured_context = student_data_to_context(
        student_data
    )

    # Convert the user query into a vector
    # so that it can be compared with document vectors.
    query_vector = embed_texts([question])[0]

    # Search ChromaDB for the most relevant
    # document chunks belonging to this student.
    results = serch_chunks(
        query_vector,
        student_id=student_data["student_id"],
        n_results=3,
    )

    # Extract retrieved documents and metadata
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Combine the retrieved document chunks
    # into one piece of context.
    document_context = "\n\n".join(documents)

    # Combine structured student data
    # and unstructured document data.
    context = f"""
STRUCTURED STUDENT DATA:

{structured_context}

UNSTRUCTURED DOCUMENT DATA:

{document_context}
"""

    # Send the question and combined context
    # to Gemini to generate the final answer.
    response = generate_answer(
        question,
        context,
    )

    # Collect document sources
    sources = []

    for metadata in metadatas:
        source = metadata.get("source")

        if source:
            sources.append({
                "document_id": metadata.get("document_id"),
                "source": source,
                "chunk_index": metadata.get("chunk_index"),
            })

    # Add sources to the generated response
    response["sources"] = sources

    return response

