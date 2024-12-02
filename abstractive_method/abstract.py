from transformers import BartForConditionalGeneration, BartTokenizer

# Using Bart as it is a good for generating text by AI

def abstractive_summary(text, model_name="facebook/bart-large-cnn", max_length=130, min_length=30):
    # Load pre-trained BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Tokenizes the input text in able to easier summarize and generate text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generates the summary from the generated tokens
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decodes the generated tokens into text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# # Example text
# example_text = """
# The University of Missouri football program, commonly referred to as Mizzou Football, 
# has a rich and storied history dating back to its founding in 1890. Representing the 
# University of Missouri in the NCAA Division I Football Bowl Subdivision (FBS), the 
# Missouri Tigers have been a formidable presence in college football for well over a 
# century. The team plays its home games at Faurot Field, part of Memorial Stadium, 
# located in Columbia, Missouri. Known for its electric atmosphere and passionate fan 
# base, Faurot Field is often filled with a sea of black and gold, reflecting the team’s colors.
# """

# # Generate and print the summary
# summary = abstractive_summary(example_text)
# print("Summary:")
# print(summary)

import time

start = time.time()
i = 0
try:
    with open("wiki_articles.txt", 'r') as input_file, open("simplified.txt", 'w') as output_file:
        for line in input_file:
            output_file.write(abstractive_summary(line)+'\n')
            i += 1
except:
    print("Error Occured")
finally:
    end = time.time()
    print(f"Time elapsed: {end-start}\nArticles Processed: {i}")