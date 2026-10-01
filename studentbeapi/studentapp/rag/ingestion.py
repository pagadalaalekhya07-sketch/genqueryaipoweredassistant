from pathlib import Path
from studentapp.rag.extractor import extract_text_from_pdf
from studentapp.rag.chunker import split_text
from studentapp.rag.embeddings import embed_texts
from studentapp.rag.vector_store import (
    store_chunks,
    delete_document_chunks
)
def ingest_document(document):
    file_path = Path(document.file.path)

    if file_path.suffix.lower()!=".pdf":
        return{
            "status":"skipped",
            "message":"only pdf files are supported "
        }
    text = extract_text_from_pdf(file_path)
    #only for text
    if not text.strip():
        return{
            "status":"failed",
            "message":"no text could be exctrated from the pdf"
        }
    #for chunk
    chunks= split_text(text)
    if not chunks:
        return{
            "status":"failed",
            "message":"no chunks were created"
        }
    try:
        embeddings = embed_texts(chunks)

    except Exception as e:
        return{
            "status":"failed",
            "message":(
                f"embedding failed :{str(e)}"
            )
        }
    #make sure every chunk has its corresponding vector
    if len(embeddings) != len(chunks):
        return{
            "status":"failed",
            "message":(
                "number of embeddings doesnt match"
                "number of chunk"
            )
        }
    
    try:
        delete_document_chunks(
            document.id 
        )
    except Exception as e:
        return{
            "status":"failed",
            "message":(
                f"could not delete old chunks:{str(e)}"
            )
        }
    #make a connection
    try:
        store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            student_id=document.student_id,
            document_id=document.id,
            source=document.title
        )
    except Exception as e:
        return{
            "status":"failed",
            "message":{
                f"chromadb storage failed:{str(e)}"
            }
        }
    return{
        "status":"success",
        "document_id" : document.id,
        "student_id": document.student.id,
        "chunks":len(chunks)
    }