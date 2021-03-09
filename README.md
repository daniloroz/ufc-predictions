# UFC Predictions
   *TLDR* : Create a machine learning model to accurately predict the victorious contender in the **UFC** promotion.
   Best models are able to yield a 70 percent accuracy.
   > See Jupyter Python notebook
   
   **Tools Needed**
   
   <img src="https://github.com/Xx-Ashutosh-xX/Xx-Ashutosh-xX/blob/master/assets/icons/ai.png" alt="AI" width="60" hight="20"> <img src="https://github.com/Xx-Ashutosh-xX/Xx-Ashutosh-xX/blob/master/assets/icons/python.png" alt="python" width="80" hight="20">
   
   - Model Generation: Scikit-learn, Tensorflow
   - Data Manipulation: pandas, numpy, feature_engine
   - Documentation found [here](https://scikit-learn.org/)
   *(jupyter notebook is recommended for easiest compliation)*
   
   **To Compile**
   1. clone or download code, cd into directory. If using .ipynb follow instructions [here](https://stackoverflow.com/questions/53254703/import-its-own-ipynb-files-on-google-colab) on how to import to google collab *easiest method* (no need to install packages on local machine)
   2. install using pip
   
   ![GitHub Logo](/images/ufc.png)
## 0.1 Prelude:

  The Ultimate Fighting Championship (UFC) is currently one of the fastest-growing sports in the world (Telegraph, 2017) and organises events weekly.
  
   My first ever UFC event was UFC 241 with Stipe vs Cormier 2 fighting for the world heavyweight championship at my friends house (I fell asleep after the first round). Ever since, I have become hooked to this sport not missing a single event. The excitement,the passion, the journey to champion, the devastation of a loss; it all adds to the appeal. Soon after I joined my coworkers subreddit [r/MMApredictions](https://www.reddit.com/r/mmapredictions/), where fans can predict the fights for points (basically for bragging rights).
   
   
   
   ![GitHub Logo](/images/DC.jpg)

   With my background in computer science and Machine Learning, I thought of creating a ML model to predict with a higher accuracy than what I can currently pick. (since machine analysis can be superior to human analysis and I mostly end up guessing). 


## 1 Objective:
- Create a ML model to analyze fight data in the UFC promotion and predict the winner of the bout.
- Acheive an acceptable accuracy (most models already out the has acheived anywhere from 70%-80% some example models [here](https://github.com/WarrierRajeev/UFC-Predictions) and [here](https://github.com/rezan21/UFC-Prediction/blob/master/README.md))
- Find an Updated dataset with current roster and records
- Have some visualization generated, to give some insights on which factors plays into winning the bout
- Use package SciKit-Learn. Documentation can be found [here](https://scikit-learn.org/)

## 2 Dataset:
- With every ML project, the performance and accuracy heavily relies on the consistency of the dataset. 
- The dataset that i will be using is from kaggle which was scraped from ufc statitstics website and can be found [here](https://www.kaggle.com/rajeevw/ufcdata)
- there are many entries of fight data. The more the merrier for our model to determine correlations.
- there were 3 .csv files provided:
      1. data.csv (partial processed file aggregated from csv datasets below)
      2. raw_fighter_details.csv (contains details of each separate fighter)
      3. raw_total_fight_data.csv (data from every fight from ufc)
- The data dates back to the debut of the UFC 

Below is a small sample of the columns of data with their description:

Column Name | Description
------------ | -------------
**R_** and **B_** | prefix signifies red and blue corner fighter stats respectively
**opp** | containing columns is the average of damage done by the opponent on the fighter
**KD** | is number of knockdowns
**SIG_STR** | is no. of significant strikes 'landed of attempted'
**SIG_STR_pct** | is significant strikes percentage
**TOTAL_STR** | is total strikes 'landed of attempted'
**TD** | is no. of takedowns
**CLINCH** | is no. of significant strikes in the clinch 'landed of attempted'
**GROUND** | is no. of significant strikes on the ground 'landed of attempted'
**Stance** | is the stance of the fighter (orthodox, southpaw, etc.)
**Height_cms** | is the height in centimeter
**Reach_cmsv** | is the reach of the fighter (arm span) in centimeter
**Weight** | lbs is the weight of the fighter in pounds (lbs)
**age** | is the age of the fighter
**title_bout** | Boolean value of whether it is title fight or not
**weight_class** | is which weight class the fight is in (Bantamweight, heavyweight, Women's flyweight, etc.)
**no_of_rounds** | is the number of rounds the fight was scheduled for
**current_lose_streak** | is the count of current concurrent losses of the fighter
**current_win_streak** | is the count of current concurrent wins of the fighter
**wins** | is the number of wins in the fighter's ufc career
**losses** | is the number of losses in the fighter's ufc career
**total_rounds_fought** | is the average of total rounds fought by the fighter
**total_time_fought** | (seconds) is the count of total time spent fighting in seconds
**total_title_bouts** | is the total number of title bouts taken part in by the fighter
**win_by_Decision_Majority** | is the number of wins by majority judges decision in the fighter's ufc career
**win_by_Decision_Split** | is the number of wins by split judges decision in the fighter's ufc career

## 3 Dataset Processing
- the purpose of processing the data is to find the most valuable features which will lead to higher accuracies in the model
- the dataset included a preprocessed file, however with this project I would like to perform custom feature engineering to better understand how the model will work with the data.
-  Currently working on processing dataset/data.csv. Have implemented the following processing methods:
   1. Finding Feature Variability
   2. duplicate features and mistakes in entries
   3. Missing Values
   4. Outliers
   5. Encoding and Imputations
   
     - For missing forms of data, there were imputations made with an arbitrary number (for floats 0.0).
     - Some ages were missing. They were Imputed the missing ages with the mode of th ages.
     - Imputations and encoding can be re-evaluated later to optimize the performance of the models.
     
     > Performance of a ML model relies heavily on the consistency of the dataset. The current dataset can always be worked on and optimized. Dropped columns can be recycled and formatted into a useful feature. Only basic preprocessing and cleaning has been done. I want to get a model working first.
     
     > More analysis and work on the dataset will be made afterwords. For example, using the fight dates and calculate how long the break period between their fights. This can offer more insight if this break time affects their performance in a fight. The number of possibilities to find more features from data is endless.
   
## 4 Models
##### Logistic Regression:	 
- is found in models/model_lr1.py
- Accuracy outputted is around 69.50 percent *(Not-ideal)*.
- Currently working on evaluations of the LR model and introducing new algorithms and models.
##### Naive Bayes:
- is found in models/model_NB.py
- tested Gaussian, Complement, binomial naive bayes 
##### Neural Network:
- is found in models/model_NN.py
- multi-layer perceptron (MLP) algorithm that trains using Backpropagation. Best for classifications
##### Support Vector machine Classifier (SVM):
- is found in models/model_SVM.py

## 5 Testing
- Multiple testing methods will be used on models After extensive research multiple validation and evaluation methods will be used for assessing performance.
- These testing methods include; K-Fold cross validation, cross validation, accuracy\recall\precision tests, and F-score.

## 6 Results
- Initial scores have been returning accuracies of 69.00 - 70.00 percent. Many improvements are still left to be made on the models
- Model and testing still being done. Accuracies and performance of models will be posted soon.

| Model | Accuracy | Recall | Precision |
| :---: | :---:    | :---:  | :---:     |
| LR   | 0.6958 | 0.4312  | 0.5080 |
| NB   | 0.5889 | 0.5726 |  | 0.5882
| SVM   | 0.4919 | 0.4829  | 0.4311|
| NN   | 0.4559 | 0.4385  | 0.4158 |

## 7 Usage
- Currently working on the models accepting input to predict new matchups.
- Multiple datasets must be referenced in order ot make this happen. Since model training template must be identical all around.
- An input csv file is used to predict upcoming fights:
- UFC 259 Main card is used as input which takes place on **Saturday March 6, 2021**. Below is a table to track different predicitions made by the different models.

| Model | Contenders | Picked Winner | Actual Winner |
| :---: | :---:      | :---:         | :---:         |
| LR   | Adesanya vs Błachowicz | Błachowicz | Błachowicz |
| LR   | Nunes vs Anderson | Nunes | Nunes |
| LR   | Yan vs Sterling | Yan | DQ |
| LR   | Santos vs Rakic | Rakic | Rakic |
| NB   |  Adesanya vs Błachowicz | Adesanya | Błachowicz |
| NB   | Nunes vs Anderson | Nunes | Nunes |
| NB   | Yan vs Sterling | Yan | DQ |
| NB   | Santos vs Rakic | Rakic | Rakic |

> Yan vs Sterling ended via DQ maing Sterling the winner

## 8 Lessons Learned
- I am happy with the results with ~70 percent accuracies on optimal models
- The nature of picking a winning fights and being accurate is hard in itself. There are so many variables to analyze and determine correlations. I don't think there will be any models that can truly and accurately pick a winner in any sports context. It is essentially a 50/50 decision.
- People (humnans) have a good intuition of choosing the winners based on the performance and games that a sports fan has watched. I've seen some people reach 60% accuracy when picking winners in the UFC for a year.
- The dataset posed a lot of issues. It was imbalanced, and it took a lot of preprocessing to clean it up. I think for cases like the Neural Network and SVM classifier, the data was too complex and caused overfitting. Some columns could have been omitted from the training. Future work includes cleaning the dataset more, simplify the data columns, and removing un-needed columns.
- There can always be more work done to optimize the model; however, for now I will continue to work on other projects. I can see myself revisiting this project in the future. 
- Overall I am happy with this project, and I am glad to see an acceptable accuracy.

