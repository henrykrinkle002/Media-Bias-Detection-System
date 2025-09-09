import pandas as pd

df = pd.read_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/News_Articles/NexisArticles.csv')
df = df.drop_duplicates(subset = ["url", "title"])
df.to_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/News_Articles/Nexisarticles_5.csv', index = False)