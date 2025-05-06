def classify_document(text):
    """
    Dummy document classification for now.
    Replace with ML/NLP logic if available.
    """
    if "programming" in text.lower():
        return "Computer Science"
    elif "biology" in text.lower():
        return "Biological Sciences"
    return "General Document"
