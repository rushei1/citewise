from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_paper(title, abstract):
    input_text = f"{title}. {abstract}"

    input_text = input_text[:1024]  
    result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']