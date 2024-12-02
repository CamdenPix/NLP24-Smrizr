import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt')

def extractive_summary(text, num_sentences=3):
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Check if text has fewer sentences than requested
    if len(sentences) <= num_sentences:
        return text  # Return original text if it's too short
    
    # Calculate TF-IDF for the sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Score sentences by summing TF-IDF weights
    sentence_scores = tfidf_matrix.sum(axis=1).A1  # Flatten the matrix to a 1D array
    
    # Get indices of top-ranked sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    
    # Build the summary by selecting the top sentences
    summary = ' '.join([sentences[i] for i in top_indices])
    return summary

# # Example text to test the function
# example_text = """
# The University of Missouri football program, commonly referred to as Mizzou Football, 
# has a rich and storied history dating back to its founding in 1890. Representing the 
# University of Missouri in the NCAA Division I Football Bowl Subdivision (FBS), the 
# Missouri Tigers have been a formidable presence in college football for well over a 
# century. The team plays its home games at Faurot Field, part of Memorial Stadium, 
# located in Columbia, Missouri. Known for its electric atmosphere and passionate fan 
# base, Faurot Field is often filled with a sea of black and gold, reflecting the team’s colors.

# Mizzou Football has been a member of the Southeastern Conference (SEC) since 2012, 
# competing in the East Division. Prior to joining the SEC, the Tigers were part of the 
# Big 12 Conference and before that, the Big Eight Conference. The move to the SEC marked 
# a significant shift for the program, as it transitioned to playing against some of the best 
# teams in the nation, including Alabama, Georgia, and Florida. Despite early skepticism, 
# Missouri proved to be a competitive force in the SEC, winning the East Division title in 2013 
# and 2014 under head coach Gary Pinkel. These seasons remain highlights in the modern era of 
# Mizzou Football, showcasing the team’s ability to compete at the highest level.

# One of the hallmarks of Mizzou Football has been its ability to produce standout players 
# who have gone on to achieve success in the NFL. Notable alumni include Kellen Winslow, a Hall 
# of Fame tight end, and defensive linemen Justin Smith and Aldon Smith, both of whom made 
# significant impacts in the professional ranks. Quarterback Drew Lock, drafted by the Denver 
# Broncos in 2019, is another example of the program’s ability to develop top-tier talent.

# The program has seen its fair share of challenges as well. Competing in the SEC means facing 
# relentless competition every season, and Missouri has experienced ups and downs as it strives 
# to maintain its place among college football's elite programs. Recruiting has been a focal 
# point for the team, as securing top talent from across the country is essential to sustaining 
# success in the SEC. Head coach Eliah Drinkwitz, who took over the program in 2020, has placed 
# a strong emphasis on building a winning culture and improving recruiting pipelines in the region. 
# Drinkwitz has also embraced modern strategies, such as leveraging the transfer portal to bring 
# in experienced players who can make an immediate impact.

# The fan base, known for its unwavering loyalty, plays a significant role in the identity of 
# Mizzou Football. The tradition of the "Rock M," a large whitewashed stone "M" on the hill 
# overlooking Faurot Field, is a cherished symbol of the program. Game days in Columbia are 
# marked by tailgating, the Tiger Walk, and the playing of the university’s fight song, 
# “Every True Son.” These traditions create an unparalleled sense of community and pride among 
# students, alumni, and fans.

# In recent years, Mizzou Football has worked to solidify its place in the SEC while navigating 
# the evolving landscape of college athletics. The introduction of Name, Image, and Likeness (NIL) 
# opportunities has added a new dimension to recruiting and player retention, with Missouri being 
# proactive in helping student-athletes capitalize on these changes. The program is also investing 
# heavily in facilities and infrastructure to remain competitive, including recent renovations to 
# Memorial Stadium and expanded resources for player development.

# Looking ahead, the future of Mizzou Football is filled with potential. With a growing pool of 
# talented recruits and a commitment to excellence both on and off the field, the program aims to 
# reestablish itself as a perennial contender in the SEC. While challenges remain, the passion and 
# dedication of the Mizzou community ensure that the Tigers will continue to roar on the college 
# football stage for years to come.
# """

# # Generate and print a summary
# summary = extractive_summary(example_text, num_sentences=5)
# print("Summary:")
# print(summary)

import time

start = time.time()
i = 0
try:
    with open("wiki_articles.txt", 'r') as input_file, open("simplified.txt", 'w') as output_file:
        for line in input_file:
            output_file.write(extractive_summary(line)+'\n')
            i += 1
except:
    print("Error Occured")
finally:
    end = time.time()
    print(f"Time elapsed: {end-start}\nArticles Processed: {i}")