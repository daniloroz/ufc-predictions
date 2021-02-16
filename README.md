# UFC Predictions
   *TLDR* : Create a machine learning model to accurately predict the victorious contender in the **UFC** promotion.
   Best models are able to yield a 70 percent accuracy.
   > There will be a jupyter notebook file committed for simple compilation if you want to use the models yourselves
   
   **Tools Needed**
   
   <img src="https://github.com/Xx-Ashutosh-xX/Xx-Ashutosh-xX/blob/master/assets/icons/ai.png" alt="AI" width="60" hight="20"> <img src="https://github.com/Xx-Ashutosh-xX/Xx-Ashutosh-xX/blob/master/assets/icons/python.png" alt="python" width="80" hight="20">
   
   - Model Genreation: Scikit-learn, Tensorflow
   - Data Manipulation: pandas, numpy, feature_engine
   *(jupyter notebook is recommended for easiest compliation)*
   
   **To Compile**
   1. clone or download code, cd into directory. If using .ipynb follow instructions [here](https://stackoverflow.com/questions/53254703/import-its-own-ipynb-files-on-google-colab) on how to import to google collab *easiest method* (no need to install packages on local machine)
   2. install using pip
   
   ![GitHub Logo](/images/ufc.png)
## 0.1 Prelude:
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

Below is a sample of the columns of data with their description:
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
**win_by_Decision_Unanimous** | is the number of wins by unanimous judges decision in the fighter's ufc career
**win_by_KO/TKO** | is the number of wins by knockout in the fighter's ufc career
**win_by_Submission** | is the number of wins by submission in the fighter's ufc career
**win_by_TKO_Doctor_Stoppage** | is the number of wins by doctor stoppage in the fighter's ufc career

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

## 5 Testing
- Multiple testing methods will be used on models After extensive research multiple validation and evaluation methods will be used for assessing performance.
- These testing methods include; K-Fold cross validation, cross validation, accuracy\recall\precision tests, and F-score.

## 6 Results
- Initial scores have been returning accuracies of 69.00 - 70.00 percent. Many improvements are still left to be made on the models
- Model and testing still being done *(stay tuned)*
   

