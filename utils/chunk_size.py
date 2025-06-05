def get_dynamic_chunk_size(documents):
    total_words=sum(len(doc.page_content.split()) for doc in documents)

    if total_words < 1000:
        return 300  # Smaller chunks for tiny docs
    elif total_words < 3000:
        return 700
    elif total_words < 10000:
        return 1000
    else:
        return 2000  # Cap for large docs