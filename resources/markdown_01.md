## Idea behind the app
This process require users to input text (ideally a tweet relating to climate change), and it will be classified according to whether or not they believe in climate change. Below are information about the data source and a brief data description.
On the Exploratory Data Analysis page are insights on thought patterns of the colated tweets
The Model information page, predictions can be made with our already trained models.

## Data description as per source
The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.

This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).

Each tweet is labelled as one of the following classes:

2(News): the tweet links to factual news about climate change
1(Pro): the tweet supports the belief of man-made climate change
0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change
-1(Anti): the tweet does not believe in man-made climate change