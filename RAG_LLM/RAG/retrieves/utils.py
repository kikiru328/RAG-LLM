def merge_docs(docs):
    """Merge all retrieved documents into string"""
    return "\n\n".join([d.page_content for d in docs])
