def classify_document(text):
    """
    Classifies the document content based on its text.
    This is just an example classification logic. You can modify it according to your needs.

    :param text: The text content of the document
    :return: A string indicating the document's classification
    """
    if "course content" in text.lower():
        return "Course Content Document"
    elif "assessment" in text.lower():
        return "Assessment Document"
    else:
        return "Unclassified Document"
