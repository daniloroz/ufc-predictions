# UFC Predictions
   *TLDR* : Create a machine learning model to accurately predict the victorious contender in the **UFC** promotion.
   ![GitHub Logo](/images/ufc.png)
## 0.1 Prelude:
   My first ever UFC event was UFC 241 with Stipe vs Cormier 2 fighting for the world heavyweight championship at my friends house (I fell asleep after the first round). Ever since, I have become hooked to this sport not missing a single event. The excitement,the passion, the journey to champion, the devastation of a loss; it all adds to the appeal. Soon after I joined my coworkers subreddit [r/MMApredictions](https://www.reddit.com/r/mmapredictions/), where fans can predict the fights for points (basically for bragging rights).
   
   ![GitHub Logo](/images/DC.jpg)

   With my background in computer science and Machine Learning, I thought of creating a ML model to predict with a higher accuracy than what I can currently pick. (since machine analysis can be superior to human analysis and I mostly end up guessing). 


## 1 Objective:
- Create a ML model to analyze fight data in the UFC promotion and predict the winner of the bout.
- Acheive an acceptable accuracy (most models already out the has acheived anywhere from 70%-80% some example models [here](https://github.com/WarrierRajeev/UFC-Predictions) and [here](https://github.com/rezan21/UFC-Prediction/blob/master/README.md))
- Find an Updated dataset with current roster and records
- Have some visualization generated
- Have some insights on which factors plays into winning the bout

## 2 Dataset:
- With every ML project, the performance and accuracy heavily relies on the consistency of the dataset. 
- The dataset that i will be using is a from kaggle and can be found [here](https://www.kaggle.com/rajeevw/ufcdata)
- there are many entries of fight data. The more the merrier for our model to determine correlations.
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

### Currently working on dataset processing.
